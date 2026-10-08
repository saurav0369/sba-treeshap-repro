#!/usr/bin/env python3
"""Frozen, as-of-mature SBA 7(a) TreeSHAP selector experiment.

Reproducible experimental harness, NOT a report of real-data results.
Requires real SBA CSVs or --smoke for a synthetic protocol test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score
from scipy.spatial.distance import jensenshannon

NUMERIC = ["GrossApproval", "SBAGuaranteedApproval", "TermInMonths", "JobsSupported"]
CATEGORICAL = ["ProjectState", "BusinessAge", "BusinessType", "ProcessingMethod", "RevolverStatus", "Naics2"]
TARGET_DAYS = None  # precisely 36 calendar months via DateOffset
COHORTS = [
    dict(id="T2005", cutoff="2005-12-31", train_end=1998, hist=(1999, 2000), val=(2001, 2002), future=(2006, 2007)),
    dict(id="T2006", cutoff="2006-12-31", train_end=1999, hist=(2000, 2001), val=(2002, 2003), future=(2007, 2008)),
    dict(id="T2007", cutoff="2007-12-31", train_end=2000, hist=(2001, 2002), val=(2003, 2004), future=(2008, 2009)),
]
EPSILONS = (0.01, 0.02, 0.05)
SEED = 20261008


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = ["FirstDisbursementDate", "ChargeOffDate", "LoanStatus", "NaicsCode"] + NUMERIC + CATEGORICAL[:-1]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError("Missing SBA FOIA columns: " + ", ".join(missing))
    df = raw.copy()
    for c in ("FirstDisbursementDate", "ChargeOffDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    df["LoanStatus"] = df["LoanStatus"].astype(str).str.strip().str.upper()
    df = df.loc[df["FirstDisbursementDate"].notna()].copy()
    # Some CHGOFF rows lack an event date; exclude them rather than falsely code as negatives.
    incomplete = (df["LoanStatus"] == "CHGOFF") & df["ChargeOffDate"].isna()
    n_incomplete = int(incomplete.sum())
    df = df.loc[~incomplete].copy()
    bad_dates = df["ChargeOffDate"].notna() & (df["ChargeOffDate"] < df["FirstDisbursementDate"])
    n_bad_dates = int(bad_dates.sum())
    df = df.loc[~bad_dates].copy()
    df["horizon_end"] = df["FirstDisbursementDate"] + pd.DateOffset(months=36)
    df["y"] = (df["ChargeOffDate"].notna() & (df["ChargeOffDate"] <= df["horizon_end"])).astype(int)
    df["year"] = df["FirstDisbursementDate"].dt.year
    df["Naics2"] = df["NaicsCode"].fillna("missing").astype(str).str[:2]
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].fillna("MISSING").astype(str).replace({"nan": "MISSING", "": "MISSING"})
    counts = df.groupby("year")["y"].agg(["count", "sum"]).reset_index().to_dict("records")
    qa = dict(rows=int(len(df)), missing_chargeoff_dates=n_incomplete, invalid_chargeoff_before_disbursement=n_bad_dates, counts_by_year=counts,
              definition="chargeoff within 36 calendar months of first disbursement, including otherwise open loans with full 36-month observability")
    return df, qa


def make_preprocessor(train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=False), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=18), CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)
    prep.fit(train)
    names = list(prep.get_feature_names_out())
    # Semantic parent for each one-hot; numeric features remain individual.
    parents = []
    for name in names:
        if name in NUMERIC:
            parents.append(name)
        else:
            matches = [c for c in CATEGORICAL if name.startswith(c + "_")]
            if len(matches) != 1:
                raise ValueError("Cannot map encoded feature to one semantic feature: " + name)
            parents.append(matches[0])
    return prep, names, parents


def matrix(prep, x):
    a = np.asarray(prep.transform(x), dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("Features contain nonfinite values after imputation")
    return a


def config_grid(n: int) -> list[dict]:
    from itertools import product
    grid = [dict(max_depth=d, n_estimators=e, learning_rate=lr, min_child_weight=m, subsample=s,
                 colsample_bytree=c, reg_lambda=reg)
            for d, e, lr, m, s, c, reg in product(
                (2, 3, 5), (70, 150, 250), (0.025, 0.065, 0.12), (1, 5, 12), (0.75, 1.0), (0.75, 1.0), (1.0, 5.0))]
    ids = np.random.default_rng(SEED).permutation(len(grid))[:n]
    return [grid[int(i)] for i in ids]


def explain(model, background, xs):
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional", model_output="raw")
    sv = np.asarray(explainer.shap_values(xs, check_additivity=False), dtype=float)
    if sv.ndim != 2:
        raise ValueError("Expected 2-D binary-class raw-margin TreeSHAP")
    raw = model.predict(xs, output_margin=True)
    bias = float(np.asarray(explainer.expected_value).ravel()[0])
    max_error = float(np.max(np.abs(sv.sum(axis=1) + bias - raw)))
    if max_error > 0.02:
        raise AssertionError("SHAP additivity check failed: " + str(max_error))
    return sv, max_error


def correlated_groups(training: pd.DataFrame, threshold=0.8) -> dict[str, str]:
    r = training[NUMERIC].corr(method="spearman").abs()
    parent = {k: k for k in NUMERIC}
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i, u in enumerate(NUMERIC):
        for v in NUMERIC[i+1:]:
            if np.isfinite(r.loc[u, v]) and r.loc[u, v] >= threshold:
                ru, rv = root(u), root(v)
                parent[max(ru, rv)] = min(ru, rv)
    return {k: root(k) for k in NUMERIC}


def attribution_profile(sv: np.ndarray, parents: list[str], cor: dict[str, str] | None = None):
    # Sum signed SHAP within semantic groups before taking magnitude.
    groups = [cor.get(p, p) if cor else p for p in parents]
    uniq = sorted(set(groups))
    raw = np.stack([sv[:, [j for j, p in enumerate(groups) if p == g]].sum(axis=1) for g in uniq], axis=1)
    importance = np.mean(np.abs(raw), axis=0)
    if np.sum(importance) == 0:
        return np.full(len(uniq), 1 / len(uniq))
    return importance / np.sum(importance)


def agreement(profile_mat: np.ndarray, metric: str) -> np.ndarray:
    n = len(profile_mat)
    out = np.zeros(n)
    for i in range(n):
        sims = []
        a = profile_mat[i]
        for j in range(n):
            if j == i:
                continue
            b = profile_mat[j]
            if metric == "cosine":
                s = float(np.dot(a, b) / max(1e-15, np.linalg.norm(a)*np.linalg.norm(b)))
            elif metric == "jsd":
                s = float(1.0 - jensenshannon(a, b, base=2.0)**2)
            else:
                raise ValueError(metric)
            sims.append(s)
        out[i] = np.mean(sims)
    return out


def evaluate_selectors(loss_val, loss_hist, loss_fut, pv, profiles, profiles_corr=None, path_profiles=None):
    idx_val = int(np.argmin(loss_val))
    result = []
    for eps in EPSILONS:
        R = np.where(loss_val <= np.min(loss_val) * (1 + eps))[0]
        if len(R) < 3:
            result.append({"epsilon": eps, "rashomon": int(len(R)), "status": "too_few_models"})
            continue
        oracle = float(np.min(loss_fut[R]))
        sub = np.stack([profiles[i] for i in R])
        # Column-wise Pearson agreement in predictions (same validation rows for all models).
        pred = np.stack([pv[i] for i in R])
        pred_sim = np.corrcoef(pred)
        pred_sim = np.nan_to_num(pred_sim, nan=0)
        pred_agreement = (pred_sim.sum(axis=1)-1)/(len(R)-1)
        def regret(local_idx):
            return float(loss_fut[R[int(local_idx)]] - oracle)
        scores = {"validation": -loss_val[R], "history_trend": -(loss_val[R] - loss_hist[R]),
                  "prediction_agreement": pred_agreement, "ecs_cosine": agreement(sub, "cosine"),
                  "ecs_jsd": agreement(sub, "jsd")}
        for key, pp in (("correlated_group_cosine", profiles_corr), ("path_dependent_cosine", path_profiles)):
            if pp is not None:
                scores[key] = agreement(np.stack([pp[i] for i in R]), "cosine")
        answers = {k: regret(np.argmax(v)) for k, v in scores.items()}
        answers["random_expected"] = float(np.mean(loss_fut[R] - oracle))
        result.append({"epsilon": eps, "rashomon": int(len(R)), "selected_regret": answers,
                       "rank_correlation_ecs_robustness": float(pd.Series(scores["ecs_cosine"]).corr(pd.Series(-(loss_fut[R]-loss_val[R])), method="spearman"))})
    return result


def env_run(df, config, n_models, n_bg, n_explain, path_sensitivity):
    T = pd.Timestamp(config["cutoff"])
    train = df.loc[df.year <= config["train_end"]].copy()
    hist = df.loc[df.year.between(*config["hist"])].copy()
    val = df.loc[df.year.between(*config["val"])].copy()
    future = df.loc[df.year.between(*config["future"])].copy()
    # No outcome may enter selection without a completed 36-month follow-up at T.
    for name, block in (("train", train), ("history", hist), ("validation", val)):
        if (block["horizon_end"] > T).any():
            raise AssertionError(f"AS-OF LABEL LEAK in {name}, {config['id']}")
    if len(future) == 0 or (future.FirstDisbursementDate <= T).any():
        raise AssertionError("Future cohort absent or not strictly after cutoff")
    for name, block in (("train", train), ("history", hist), ("validation", val), ("future", future)):
        if len(block) < 100 or block.y.nunique() != 2:
            raise ValueError(f"Insufficient observations/events in {config['id']} {name}: {len(block)}")
    pre, names, parents = make_preprocessor(train)
    XX = [matrix(pre, x) for x in (train, hist, val, future)]
    xt, xh, xv, xf = XX
    rng = np.random.default_rng(SEED)
    # Draw once, reuse across all models (the crucial collaborator constraint).
    bi = np.sort(rng.choice(len(xt), size=min(n_bg,len(xt)), replace=False))
    vi = np.sort(rng.choice(len(xv), size=min(n_explain,len(xv)), replace=False))
    background, explanation = xt[bi], xv[vi]
    corr = correlated_groups(train)
    results = []
    for i, conf in enumerate(config_grid(n_models)):
        mdl = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", enable_categorical=False,
                                n_jobs=2, random_state=SEED+i, **conf)
        mdl.fit(xt, train.y.to_numpy())
        pv = mdl.predict_proba(xv)[:,1]
        ph = mdl.predict_proba(xh)[:,1]
        pf = mdl.predict_proba(xf)[:,1]
        sv, err = explain(mdl, background, explanation)
        primary = attribution_profile(sv, parents)
        correlated = attribution_profile(sv, parents, corr)
        path = None
        if path_sensitivity:
            path_exp = shap.TreeExplainer(mdl, feature_perturbation="tree_path_dependent", model_output="raw")
            path = attribution_profile(np.asarray(path_exp.shap_values(explanation), dtype=float), parents)
        results.append(dict(i=i, config=conf, loss_val=float(log_loss(val.y, pv)), loss_hist=float(log_loss(hist.y, ph)),
                            loss_future=float(log_loss(future.y, pf)), future_auc=float(roc_auc_score(future.y,pf)),
                            val_preds=pv, primary=primary, correlated=correlated, path=path, max_shap_additivity_error=err))
    selector_results = evaluate_selectors(
        np.asarray([r["loss_val"] for r in results]), np.asarray([r["loss_hist"] for r in results]),
        np.asarray([r["loss_future"] for r in results]), [r["val_preds"] for r in results],
        [r["primary"] for r in results], [r["correlated"] for r in results],
        [r["path"] for r in results] if path_sensitivity else None,
    )
    return dict(environment=config, sizes={k:len(v) for k,v in (("train",train),("history",hist),("validation",val),("future",future))},
                event_counts={k:int(v.y.sum()) for k,v in (("train",train),("history",hist),("validation",val),("future",future))},
                model_count=len(results), feature_count=len(names), background_rows=len(bi), explanation_rows=len(vi),
                correlated_groups={k:v for k,v in corr.items() if k!=v},
                max_shap_additivity_error=max(r["max_shap_additivity_error"] for r in results),
                selectors=selector_results, hyperparameter_grid=[r["config"] for r in results])


def make_synthetic():
    # Protocol smoke test only: synthetic timestamps and outcomes are not paper evidence.
    rng=np.random.default_rng(SEED)
    out=[]
    for year in range(1991,2011):
        for i in range(135):
            date=pd.Timestamp(year=year, month=6, day=15)
            amount=float(rng.lognormal(10.4,.7)); jobs=int(rng.poisson(5)); term=int(rng.choice([36,60,84,120,180]))
            logit=-2.0 + .18*np.log1p(amount/10000) - .08*jobs + .2*(year>=2005)
            y=int(rng.random() < 1/(1+np.exp(-logit)))
            co=(date+pd.DateOffset(months=int(rng.integers(4,34)))) if y else pd.NaT
            out.append(dict(FirstDisbursementDate=date,ChargeOffDate=co,LoanStatus="CHGOFF" if y else "EXEMPT",
                            GrossApproval=amount,SBAGuaranteedApproval=amount*.75,TermInMonths=term,JobsSupported=jobs,
                            ProjectState=rng.choice(["CA","TX","NY","FL"]),BusinessAge=rng.choice(["New","Existing"]),
                            BusinessType=rng.choice(["CORPORATION","INDIVIDUAL"]),ProcessingMethod=rng.choice(["PLP","SBA Express"]),
                            RevolverStatus=rng.choice(["0","1"]),NaicsCode=rng.choice(["541110","722511","238220"])))
    return pd.DataFrame(out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", help="Real SBA 7(a) FOIA CSV files (by decade)")
    p.add_argument("--output", default="treeshap_results.json")
    p.add_argument("--n-models", type=int, default=150)
    p.add_argument("--background-rows", type=int, default=256)
    p.add_argument("--explanation-rows", type=int, default=512)
    p.add_argument("--path-sensitivity", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Synthetic protocol validation; NOT publishable evidence")
    args=p.parse_args()
    if not args.smoke and not args.csv:
        p.error("Provide --csv real_file1.csv real_file2.csv, or explicitly use --smoke")
    if args.smoke:
        raw=make_synthetic()
    else:
        raw=pd.concat([pd.read_csv(path, low_memory=False, encoding="latin1") for path in args.csv],ignore_index=True)
    # AS_OF_APPROVAL_DISB_GUARD
    # Exclude chronologically impossible records before model fitting.
    if not args.smoke:
        required = {"Program", "ApprovalDate", "FirstDisbursementDate"}
        if not required.issubset(raw.columns):
            raise ValueError("Required SBA audit columns are missing")

        program = raw["Program"].astype(str).str.strip().str.upper()
        if not program.eq("7A").all():
            raise ValueError("Non-7(a) records detected")

        approval = pd.to_datetime(
            raw["ApprovalDate"], errors="coerce", format="mixed"
        )
        disbursement = pd.to_datetime(
            raw["FirstDisbursementDate"], errors="coerce", format="mixed"
        )

        invalid = (
            approval.notna()
            & disbursement.notna()
            & (disbursement < approval)
        )

        print("Excluded invalid chronology:", int(invalid.sum()), flush=True)
        raw = raw.loc[~invalid].copy()

    df, qa=prepare(raw)
    out={"study":"cross-model explanation agreement vs prospective future regret", "smoke_only":bool(args.smoke),
         "source_csvs":args.csv or [], "qa":qa, "epsilons":EPSILONS, "seed":SEED,
         "shap_method":"fixed-background interventional, raw margin", "background_rows_target":args.background_rows,
         "explain_rows_target":args.explanation_rows,
         "note":"Samples across cutoffs share calendar vintages: NOT independent economic environments"}
    out["environments"]=[env_run(df,c,args.n_models,args.background_rows,args.explanation_rows,args.path_sensitivity) for c in COHORTS]
    pth=Path(args.output);pth.parent.mkdir(parents=True,exist_ok=True)
    pth.write_text(json.dumps(out,indent=2,default=str))
    print(json.dumps({"output":str(pth),"smoke_only":args.smoke,
                      "environments":[{"id":e["environment"]["id"], "models":e["model_count"],
                                       "rashomon_at_2pct":e["selectors"][1]["rashomon"],
                                       "regrets_at_2pct":e["selectors"][1].get("selected_regret")} for e in out["environments"]]},indent=2))

if __name__=="__main__":
    main()


