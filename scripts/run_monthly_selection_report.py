#!/usr/bin/env python3
"""M7: 研究版月度 Top-K 推荐报告。

本脚本消费 M2 canonical dataset，沿用 M6 watchlist 口径
``U2_risk_sane + M6_xgboost_rank_ndcg``，对最新可买信号月做 full-fit
打分，并生成研究报告与推荐名单。输出不是生产持仓，也不会写入
promoted registry 或交易指令。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import slugify_token
from scripts.run_monthly_selection_baselines import (
    EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
    POOL_RULES,
    _format_markdown_table,
    _json_sanitize,
    load_baseline_dataset,
    summarize_candidate_pool_reject_reason,
    summarize_candidate_pool_width,
)
from scripts.run_monthly_selection_ltr import (
    M6RunConfig,
    _tag_importance,
    _train_predict_xgboost_ranker,
    build_m6_feature_spec,
    summarize_ltr_feature_importance,
)
from scripts.run_monthly_selection_multisource import M5RunConfig, _cap_fit_rows, attach_enabled_families
from src.settings import load_config, resolve_config_path


@dataclass(frozen=True)
class M7RunConfig:
    top_ks: tuple[int, ...] = (20, 30)
    report_top_k: int = 20
    candidate_pools: tuple[str, ...] = ("U2_risk_sane",)
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    availability_lag_days: int = 30
    relevance_grades: int = 5
    model_name: str = "M6_xgboost_rank_ndcg"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成月度选股 M7 研究版推荐报告")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_m7_recommendation_report")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--signal-date", type=str, default="", help="报告信号日；留空时取最新 candidate_pool_pass 月份。")
    p.add_argument("--top-k", type=str, default="20,30")
    p.add_argument("--report-top-k", type=int, default=20)
    p.add_argument("--candidate-pools", type=str, default="U2_risk_sane")
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument("--max-fit-rows", type=int, default=0)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--availability-lag-days", type=int, default=30)
    p.add_argument("--relevance-grades", type=int, default=5)
    p.add_argument(
        "--families",
        type=str,
        default="industry_breadth,fund_flow,fundamental",
        help="M7 沿用 M6 主输入特征家族。",
    )
    p.add_argument(
        "--evidence-stem",
        type=str,
        default="",
        help="M6 历史证据 stem；留空时自动寻找最新 monthly_selection_m6_ltr_*。",
    )
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _parse_int_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def select_report_signal_date(
    dataset: pd.DataFrame,
    *,
    candidate_pools: tuple[str, ...],
    requested: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    df = dataset[dataset["candidate_pool_version"].isin(candidate_pools)].copy()
    if requested:
        target = pd.Timestamp(requested).normalize()
        part = df[df["signal_date"] == target]
        if part.empty:
            raise ValueError(f"报告信号日不存在于 dataset: {target.date()}")
        if not part["candidate_pool_pass"].astype(bool).any():
            raise ValueError(f"报告信号日无 candidate_pool_pass 标的: {target.date()}")
        return target
    eligible = df[df["candidate_pool_pass"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("没有可用于 M7 推荐的 candidate_pool_pass 信号月。")
    return pd.Timestamp(eligible["signal_date"].max()).normalize()


def build_full_fit_report_scores(
    dataset: pd.DataFrame,
    spec: Any,
    cfg: M7RunConfig,
    *,
    report_signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in spec.feature_cols if c in dataset.columns]
    if not feature_cols:
        return pd.DataFrame(), pd.DataFrame()
    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for pool in cfg.candidate_pools:
        pool_df = dataset[dataset["candidate_pool_version"] == pool].copy()
        train = pool_df[
            pool_df["candidate_pool_pass"].astype(bool)
            & (pool_df["signal_date"] < report_signal_date)
            & pool_df[LABEL_COL].notna()
        ].copy()
        test = pool_df[
            pool_df["candidate_pool_pass"].astype(bool) & (pool_df["signal_date"] == report_signal_date)
        ].copy()
        if test.empty:
            warnings.warn(f"M7 pool={pool} signal_date={report_signal_date.date()} 无可推荐标的。", RuntimeWarning)
            continue
        if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows:
            warnings.warn(
                f"M7 pool={pool} 训练窗不足：months={train['signal_date'].nunique()} rows={len(train)}",
                RuntimeWarning,
            )
            continue
        train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)
        print(
            "[monthly-m7] "
            f"pool={pool} report_signal_date={report_signal_date.date()} "
            f"train_months={train['signal_date'].nunique()} train_rows={len(train_fit)} target_rows={len(test)}",
            flush=True,
        )
        scores, imp = _train_predict_xgboost_ranker(
            model_name=cfg.model_name,
            objective="rank:ndcg",
            train=train_fit,
            test=test,
            feature_cols=feature_cols,
            random_seed=cfg.random_seed,
            relevance_grades=cfg.relevance_grades,
        )
        if scores is not None and not scores.empty:
            scores = scores.copy()
            scores["feature_spec"] = spec.name
            scores["feature_families"] = ",".join(spec.families)
            scores["score_percentile"] = pd.to_numeric(scores["score"], errors="coerce")
            score_frames.append(scores)
        if imp is not None and not imp.empty:
            importance_frames.append(_tag_importance(imp, spec, pool, report_signal_date))

    scores_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return scores_out, imp_out


def _display_symbol(symbol: Any) -> str:
    raw = str(symbol)
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.zfill(6) if digits else raw


def _name_column(df: pd.DataFrame) -> pd.Series:
    for col in ["name", "stock_name", "股票名称", "名称"]:
        if col in df.columns:
            names = df[col].fillna("").astype(str).str.strip()
            return names.mask(names.eq("") | names.str.lower().eq("nan"), "UNKNOWN")
    return pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype=object)


def summarize_report_feature_coverage(
    dataset: pd.DataFrame,
    spec: Any,
    *,
    candidate_pools: tuple[str, ...],
) -> pd.DataFrame:
    base = dataset[dataset["candidate_pool_version"].isin(candidate_pools)].copy()
    if base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    pool_pass = base["candidate_pool_pass"].astype(bool)
    for pool, pool_df in base.groupby("candidate_pool_version", sort=True):
        pool_pass_part = pool_df["candidate_pool_pass"].astype(bool)
        for col in spec.feature_cols:
            raw_col = col[:-2] if col.endswith("_z") else col
            vals = (
                pd.to_numeric(pool_df[raw_col], errors="coerce")
                if raw_col in pool_df.columns
                else pd.Series(np.nan, index=pool_df.index)
            )
            rows.append(
                {
                    "candidate_pool_version": pool,
                    "feature_spec": spec.name,
                    "families": ",".join(spec.families),
                    "feature": col,
                    "raw_feature": raw_col,
                    "rows": int(len(pool_df)),
                    "candidate_pool_pass_rows": int(pool_pass_part.sum()),
                    "non_null": int(vals.notna().sum()),
                    "coverage_ratio": float(vals.notna().mean()) if len(pool_df) else np.nan,
                    "candidate_pool_pass_coverage_ratio": float(vals.loc[pool_pass_part].notna().mean())
                    if pool_pass_part.any()
                    else np.nan,
                    "first_signal_date": str(pool_df.loc[vals.notna(), "signal_date"].min().date())
                    if vals.notna().any()
                    else "",
                    "last_signal_date": str(pool_df.loc[vals.notna(), "signal_date"].max().date())
                    if vals.notna().any()
                    else "",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty and pool_pass.any():
        return out
    return out


def _build_previous_rank_map(
    previous_holdings: pd.DataFrame,
    *,
    report_signal_date: pd.Timestamp,
    pool: str,
    model: str,
    top_k: int,
) -> dict[str, int]:
    if previous_holdings.empty:
        return {}
    prev = previous_holdings.copy()
    prev["signal_date"] = pd.to_datetime(prev["signal_date"], errors="coerce").dt.normalize()
    prev = prev[
        (prev["signal_date"] < report_signal_date)
        & (prev["candidate_pool_version"] == pool)
        & (prev["model"] == model)
        & (pd.to_numeric(prev["top_k"], errors="coerce") == int(top_k))
    ].copy()
    if prev.empty:
        return {}
    last_date = prev["signal_date"].max()
    prev = prev[prev["signal_date"] == last_date].copy()
    rank_col = "selected_rank" if "selected_rank" in prev.columns else "rank"
    return {
        _display_symbol(row["symbol"]): int(row[rank_col])
        for _, row in prev.iterrows()
        if pd.notna(row.get(rank_col))
    }


def _feature_contrib_text(row: pd.Series, imp: pd.DataFrame, feature_cols: list[str], *, max_features: int = 3) -> str:
    if imp.empty:
        return ""
    model = row.get("model", "")
    pool = row.get("candidate_pool_version", "")
    view = imp[(imp["model"] == model) & (imp["candidate_pool_version"] == pool)].copy()
    if view.empty:
        view = imp[imp["model"] == model].copy()
    if view.empty:
        return ""
    imp_map = view.groupby("feature")["importance"].mean().to_dict()
    rows: list[tuple[float, str]] = []
    for feature in feature_cols:
        if feature not in row.index or feature not in imp_map:
            continue
        val = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
        importance = float(imp_map.get(feature, 0.0) or 0.0)
        if not np.isfinite(val) or not np.isfinite(importance) or importance <= 0:
            continue
        rows.append((abs(float(val)) * importance, f"{feature}={float(val):+.3g}"))
    rows.sort(key=lambda x: x[0], reverse=True)
    return "; ".join(text for _, text in rows[:max_features])


def _buyability_text(row: pd.Series) -> str:
    ok = bool(row.get("is_buyable_tplus1_open", False))
    if ok:
        return "buyable_tplus1_open"
    reason = str(row.get("buyability_reject_reason", "") or "").strip()
    return reason or "not_buyable_tplus1_open"


def _risk_flags_text(row: pd.Series) -> str:
    flags: list[str] = []
    for col in ["risk_flags", "candidate_pool_reject_reason"]:
        val = str(row.get(col, "") or "").strip()
        if val and val.lower() != "nan":
            flags.extend([x.strip() for x in val.split(";") if x.strip()])
    if not bool(row.get("is_buyable_tplus1_open", False)):
        reason = str(row.get("buyability_reject_reason", "") or "").strip()
        flags.append(reason or "not_buyable_tplus1_open")
    return ";".join(dict.fromkeys(flags))


def build_recommendation_table(
    scores: pd.DataFrame,
    dataset: pd.DataFrame,
    feature_importance: pd.DataFrame,
    *,
    feature_cols: list[str],
    top_ks: list[int],
    previous_holdings: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    previous = previous_holdings if previous_holdings is not None else pd.DataFrame()
    keys = ["signal_date", "candidate_pool_version", "symbol"]
    feature_keep = [
        c
        for c in [
            *keys,
            *feature_cols,
            "name",
            "stock_name",
            "股票名称",
            "名称",
            "industry_level1",
            "industry_level2",
            "candidate_pool_rule",
            "candidate_pool_reject_reason",
            "buyability_reject_reason",
        ]
        if c in dataset.columns
    ]
    enriched = scores.merge(dataset[feature_keep], on=keys, how="left", suffixes=("", "_dataset"))
    if "industry_level1_dataset" in enriched.columns:
        enriched["industry_level1"] = enriched["industry_level1"].fillna(enriched["industry_level1_dataset"])
    if "industry_level2_dataset" in enriched.columns:
        enriched["industry_level2"] = enriched["industry_level2"].fillna(enriched["industry_level2_dataset"])
    enriched["name"] = _name_column(enriched)

    rec_frames: list[pd.DataFrame] = []
    contrib_rows: list[dict[str, Any]] = []
    for (signal_date, pool, model), part in enriched.groupby(
        ["signal_date", "candidate_pool_version", "model"], sort=True
    ):
        ranked = part.sort_values(["score", "symbol"], ascending=[False, True]).copy()
        ranked["rank_all"] = np.arange(1, len(ranked) + 1)
        for k in top_ks:
            top = ranked.head(int(k)).copy()
            prev_map = _build_previous_rank_map(
                previous,
                report_signal_date=pd.Timestamp(signal_date),
                pool=str(pool),
                model=str(model),
                top_k=int(k),
            )
            rows: list[dict[str, Any]] = []
            for i, (_, row) in enumerate(top.iterrows(), start=1):
                symbol = _display_symbol(row["symbol"])
                contrib = _feature_contrib_text(row, feature_importance, feature_cols)
                rows.append(
                    {
                        "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                        "top_k": int(k),
                        "rank": int(i),
                        "symbol": symbol,
                        "name": str(row.get("name", "") or ""),
                        "score": float(row.get("score")) if pd.notna(row.get("score")) else np.nan,
                        "score_percentile": float(row.get("score_percentile"))
                        if pd.notna(row.get("score_percentile"))
                        else np.nan,
                        "industry": str(row.get("industry_level1", "") or ""),
                        "industry_level2": str(row.get("industry_level2", "") or ""),
                        "feature_contrib": contrib,
                        "risk_flags": _risk_flags_text(row),
                        "last_month_rank": prev_map.get(symbol, pd.NA),
                        "last_month_selected": symbol in prev_map,
                        "buyability": _buyability_text(row),
                        "candidate_pool_version": str(pool),
                        "candidate_pool_rule": str(row.get("candidate_pool_rule", POOL_RULES.get(str(pool), "")) or ""),
                        "model": str(model),
                        "model_type": str(row.get("model_type", "") or ""),
                        "feature_spec": str(row.get("feature_spec", "") or ""),
                    }
                )
                contrib_rows.append(
                    {
                        "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                        "top_k": int(k),
                        "rank": int(i),
                        "symbol": symbol,
                        "candidate_pool_version": str(pool),
                        "model": str(model),
                        "feature_contrib": contrib,
                    }
                )
            rec_frames.append(pd.DataFrame(rows))
    rec = pd.concat(rec_frames, ignore_index=True) if rec_frames else pd.DataFrame()
    if not rec.empty:
        rec["last_month_rank"] = pd.to_numeric(rec["last_month_rank"], errors="coerce").astype("Int64")
    contrib_df = pd.DataFrame(contrib_rows)
    return rec, contrib_df


def summarize_recommendation_industry_exposure(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    total = (
        recommendations.groupby(["signal_date", "candidate_pool_version", "model", "top_k"], sort=True)["symbol"]
        .nunique()
        .rename("topk_count")
        .reset_index()
    )
    out = (
        recommendations.groupby(
            ["signal_date", "candidate_pool_version", "model", "top_k", "industry"],
            dropna=False,
            sort=True,
        )["symbol"]
        .nunique()
        .rename("industry_count")
        .reset_index()
        .merge(total, on=["signal_date", "candidate_pool_version", "model", "top_k"], how="left")
    )
    out["industry_share"] = out["industry_count"] / out["topk_count"].replace(0, np.nan)
    return out


def summarize_recommendation_risk(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (signal_date, pool, model, top_k), part in recommendations.groupby(
        ["signal_date", "candidate_pool_version", "model", "top_k"], sort=True
    ):
        flag_count = int(part["risk_flags"].fillna("").astype(str).str.len().gt(0).sum())
        not_buyable = int((part["buyability"] != "buyable_tplus1_open").sum())
        rows.append(
            {
                "signal_date": signal_date,
                "candidate_pool_version": pool,
                "model": model,
                "top_k": int(top_k),
                "selected_count": int(len(part)),
                "risk_flagged_count": flag_count,
                "not_buyable_count": not_buyable,
                "last_month_selected_count": int(part["last_month_selected"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _latest_evidence_stem(results_dir: Path) -> str:
    manifests = sorted(results_dir.glob("monthly_selection_m6_ltr_*_manifest.json"))
    if not manifests:
        return ""
    return manifests[-1].name[: -len("_manifest.json")]


def _read_evidence(results_dir: Path, stem: str, suffix: str) -> pd.DataFrame:
    if not stem:
        return pd.DataFrame()
    path = results_dir / f"{stem}_{suffix}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _filter_evidence(df: pd.DataFrame, cfg: M7RunConfig) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "candidate_pool_version" in out.columns:
        out = out[out["candidate_pool_version"].isin(cfg.candidate_pools)].copy()
    if "model" in out.columns:
        out = out[out["model"] == cfg.model_name].copy()
    if "top_k" in out.columns:
        out = out[pd.to_numeric(out["top_k"], errors="coerce").isin(list(cfg.top_ks))].copy()
    return out


def build_quality_payload(
    *,
    dataset: pd.DataFrame,
    recommendations: pd.DataFrame,
    report_signal_date: pd.Timestamp,
    spec: Any,
    cfg: M7RunConfig,
    dataset_path: Path,
    db_path: Path,
    output_stem: str,
    config_source: str,
    research_config_id: str,
    evidence_stem: str,
) -> dict[str, Any]:
    target = dataset[
        dataset["candidate_pool_version"].isin(cfg.candidate_pools)
        & (dataset["signal_date"] == report_signal_date)
    ].copy()
    return {
        "result_type": "monthly_selection_m7_recommendation_report",
        "research_topic": "monthly_selection_m7_recommendation_report",
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "duckdb_path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pool_version": ",".join(cfg.candidate_pools),
        "candidate_pool_rule": {p: POOL_RULES.get(p, "") for p in cfg.candidate_pools},
        "candidate_pool_width_by_month": "see candidate_pool_width.csv",
        "feature_spec": {"name": spec.name, "families": list(spec.families), "feature_count": len(spec.feature_cols)},
        "label_spec": "full-fit ranker trained on historical forward_1m_excess_vs_market relevance; target report month has no label requirement",
        "pit_policy": "target features are signal-date rows; full-fit model uses only labeled months before report_signal_date",
        "universe_filter": "candidate_pool_pass in selected pool; no alpha filter outside model ranking",
        "industry_map_source": "industry_level1/2 columns from M2 canonical dataset",
        "data_quality_report": "M2/M5/M6 artifacts plus current M7 feature_coverage/candidate_pool_width files",
        "model_type": "xgboost_ranker_full_fit_for_research_report",
        "train_window": f"< {report_signal_date.date()} labeled signal months",
        "validation_window": f"M6 historical walk-forward evidence stem: {evidence_stem or 'not_found'}",
        "test_window": f"{report_signal_date.date()} recommendation month, unlabeled at report time",
        "cv_policy": "historical evidence uses walk_forward_by_signal_month; current report uses full-fit on past labeled months",
        "hyperparameter_policy": "fixed M6 watchlist hyperparameters; no random CV and no target-month tuning",
        "random_seed": int(cfg.random_seed),
        "rebalance_rule": "M",
        "execution_mode": "tplus1_open",
        "benchmark_return_mode": "market_ew_open_to_open",
        "top_k": list(cfg.top_ks),
        "report_top_k": int(cfg.report_top_k),
        "cost_assumption": f"{float(cfg.cost_bps):.4g} bps per unit half-L1 turnover",
        "buyability_policy": "selected rows must pass candidate_pool_pass; buyability column reports t+1 open status",
        "report_signal_date": str(report_signal_date.date()),
        "next_trade_date": str(target["next_trade_date"].dropna().iloc[0].date())
        if "next_trade_date" in target.columns and target["next_trade_date"].notna().any()
        else "",
        "target_candidate_rows": int(len(target)),
        "target_candidate_pass_rows": int(target["candidate_pool_pass"].astype(bool).sum()) if not target.empty else 0,
        "recommendation_rows": int(len(recommendations)),
        "model": cfg.model_name,
        "production_status": "research_only_not_promoted",
    }


def build_doc(
    *,
    quality: dict[str, Any],
    recommendations: pd.DataFrame,
    leaderboard: pd.DataFrame,
    risk_summary: pd.DataFrame,
    industry_exposure: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    main = recommendations[recommendations["top_k"] == quality.get("report_top_k", 20)].copy()
    main = main.sort_values(["top_k", "rank"]).head(int(quality.get("report_top_k", 20)))
    leader_view = leaderboard.sort_values(
        ["top_k", "topk_excess_after_cost_mean", "rank_ic_mean"],
        ascending=[True, False, False],
    ).head(20) if not leaderboard.empty else pd.DataFrame()
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection M7 Recommendation Report

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_m7_recommendation_report`
- 研究配置：`{quality.get('research_config_id', '')}`
- 报告信号日：`{quality.get('report_signal_date', '')}`
- 下一交易日：`{quality.get('next_trade_date', '')}`
- 候选池：`{quality.get('candidate_pool_version', '')}`
- 模型：`{quality.get('model', '')}`
- 生产状态：`research_only_not_promoted`

## 推荐名单

{_format_markdown_table(main, max_rows=int(quality.get('report_top_k', 20)))}

## M6 Historical Evidence

{_format_markdown_table(leader_view, max_rows=20)}

## Risk Summary

{_format_markdown_table(risk_summary, max_rows=20)}

## Industry Exposure

{_format_markdown_table(industry_exposure, max_rows=40)}

## Feature Coverage

{_format_markdown_table(feature_coverage.head(80), max_rows=80)}

## 口径与结论

- 本轮新增的是 M7 研究版推荐报告，不新增生产策略；推荐名单不自动生成交易指令。
- 数据质量沿用 M2/M5/M6 的 PIT 检查，本脚本只使用 `report_signal_date` 当日可观测特征，并只用该日期之前已完成标签的月份训练。
- 候选池使用 `U2_risk_sane` watchlist 口径，只做准入过滤；alpha 判断来自 `M6_xgboost_rank_ndcg` 排序分数。
- 历史 baseline 改善证据来自 M6 walk-forward 附件；M6 结论仍是 watchlist，不进入生产。
- 当前推荐月尚无未来收益标签，因此本报告不声称本月已跑赢市场；稳定性判断以历史 leaderboard / monthly_long / rank_ic / quantile_spread 为准。
- 分桶、年份和 regime 的历史失败月份保留在附件中；后续进入 regime-aware calibration 复核。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    args = parse_args()
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = str(resolve_config_path(args.config)) if args.config is not None else "default_config_lookup"
    dataset_path = _resolve_project_path(args.dataset)
    db_path_raw = args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb")
    db_path = _resolve_project_path(db_path_raw)
    results_dir_raw = args.results_dir.strip() or str(paths.get("results_dir") or "data/results")
    results_dir = _resolve_project_path(results_dir_raw)
    docs_dir = ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    top_ks = _parse_int_list(args.top_k)
    pools = tuple(_parse_str_list(args.candidate_pools))
    enabled_families = _parse_str_list(args.families)
    cfg = M7RunConfig(
        top_ks=tuple(top_ks),
        report_top_k=int(args.report_top_k),
        candidate_pools=pools,
        min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows),
        max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        relevance_grades=int(args.relevance_grades),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_model_{slugify_token(cfg.model_name)}"
        f"_maxfit_{int(args.max_fit_rows)}"
        f"_wf_{int(args.min_train_months)}m"
    )
    print(f"[monthly-m7] research_config_id={research_config_id}")

    dataset = load_baseline_dataset(dataset_path, candidate_pools=list(pools))
    m5_cfg = M5RunConfig(
        top_ks=cfg.top_ks,
        candidate_pools=cfg.candidate_pools,
        min_train_months=cfg.min_train_months,
        min_train_rows=cfg.min_train_rows,
        max_fit_rows=cfg.max_fit_rows,
        cost_bps=cfg.cost_bps,
        random_seed=cfg.random_seed,
        availability_lag_days=cfg.availability_lag_days,
    )
    dataset = attach_enabled_families(dataset, db_path, m5_cfg, enabled_families)
    spec = build_m6_feature_spec(enabled_families)
    feature_coverage = summarize_report_feature_coverage(dataset, spec, candidate_pools=cfg.candidate_pools)
    report_signal_date = select_report_signal_date(
        dataset,
        candidate_pools=cfg.candidate_pools,
        requested=args.signal_date.strip() or None,
    )
    scores, raw_importance = build_full_fit_report_scores(
        dataset,
        spec,
        cfg,
        report_signal_date=report_signal_date,
    )
    feature_importance = summarize_ltr_feature_importance(raw_importance)

    evidence_stem = args.evidence_stem.strip() or _latest_evidence_stem(results_dir)
    previous_holdings = _read_evidence(results_dir, evidence_stem, "topk_holdings")
    recommendations, feature_contrib = build_recommendation_table(
        scores,
        dataset,
        feature_importance,
        feature_cols=[c for c in spec.feature_cols if c in dataset.columns],
        top_ks=top_ks,
        previous_holdings=previous_holdings,
    )
    leaderboard = _filter_evidence(_read_evidence(results_dir, evidence_stem, "leaderboard"), cfg)
    monthly_long = _filter_evidence(_read_evidence(results_dir, evidence_stem, "monthly_long"), cfg)
    rank_ic = _filter_evidence(_read_evidence(results_dir, evidence_stem, "rank_ic"), cfg)
    quantile_spread = _filter_evidence(_read_evidence(results_dir, evidence_stem, "quantile_spread"), cfg)
    year_slice = _filter_evidence(_read_evidence(results_dir, evidence_stem, "year_slice"), cfg)
    regime_slice = _filter_evidence(_read_evidence(results_dir, evidence_stem, "regime_slice"), cfg)
    industry_exposure = summarize_recommendation_industry_exposure(recommendations)
    risk_summary = summarize_recommendation_risk(recommendations)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    quality = build_quality_payload(
        dataset=dataset,
        recommendations=recommendations,
        report_signal_date=report_signal_date,
        spec=spec,
        cfg=cfg,
        dataset_path=dataset_path,
        db_path=db_path,
        output_stem=output_stem,
        config_source=config_source,
        research_config_id=research_config_id,
        evidence_stem=evidence_stem,
    )

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "recommendations": results_dir / f"{output_stem}_recommendations.csv",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "feature_contrib": results_dir / f"{output_stem}_feature_contrib.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "risk_summary": results_dir / f"{output_stem}_risk_summary.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }

    recommendations.to_csv(paths_out["recommendations"], index=False)
    recommendations.to_csv(paths_out["topk_holdings"], index=False)
    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    feature_contrib.to_csv(paths_out["feature_contrib"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    risk_summary.to_csv(paths_out["risk_summary"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)

    summary_payload = {
        "quality": quality,
        "recommendations_top": recommendations[
            recommendations["top_k"] == int(cfg.report_top_k)
        ].head(int(cfg.report_top_k)).to_dict(orient="records")
        if not recommendations.empty
        else [],
        "historical_evidence_stem": evidence_stem,
    }
    paths_out["summary_json"].write_text(
        json.dumps(_json_sanitize(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths = [
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        **quality,
        "artifacts": [*artifact_paths, str(paths_out["doc"].relative_to(ROOT))],
    }
    paths_out["manifest"].write_text(
        json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths_out["doc"].write_text(
        build_doc(
            quality=quality,
            recommendations=recommendations,
            leaderboard=leaderboard,
            risk_summary=risk_summary,
            industry_exposure=industry_exposure,
            feature_coverage=feature_coverage,
            artifacts=[*artifact_paths, str(paths_out["manifest"].relative_to(ROOT))],
        ),
        encoding="utf-8",
    )

    print(f"[monthly-m7] report_signal_date={quality['report_signal_date']} recommendations={len(recommendations)}")
    print(f"[monthly-m7] recommendations={paths_out['recommendations']}")
    print(f"[monthly-m7] doc={paths_out['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
