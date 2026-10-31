#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对run_sb_hysteresis_20261031.py落盘结果独立复算/归因，不重新调参。"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import rotation_backtest as rb

ROOT = Path("/data/workspace/quant/sb_hysteresis_20261031")
NDX = "513100.SS"


def perf(eq):
    r = eq.pct_change().dropna()
    return {"total": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "mdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(244))}


def main():
    rb.DATA = str(ROOT / "y-data.csv")
    closes, opens = rb.load_daily()
    closes, opens = closes.loc["2013-08-01":], opens.loc["2013-08-01":]
    results = json.loads((ROOT / "sensitivity.json").read_text())
    episodes = json.loads((ROOT / "episodes.json").read_text())
    eqs = pd.read_csv(ROOT / "all_equity.csv", index_col=0, parse_dates=True)
    for r in results:
        label = r["label"]
        eq = eqs[label].dropna()
        p = perf(eq)
        assert abs(p["total"] - r["total"]) < 1e-10
        assert abs(p["mdd"] - r["mdd"]) < 1e-10
        assert abs(p["sharpe"] - r["sharpe"]) < 1e-10
        r["mdd_vs_base_pct"] = (r["mdd"] - results[0]["mdd"]) * 100
        r["total_vs_base_pct"] = (r["total"] - results[0]["total"]) * 100
        r["sharpe_vs_base"] = r["sharpe"] - results[0]["sharpe"]
        r["ndx_net_contribution_vs_base"] = r["ndx_attribution"]["net_nav_delta"] - results[0]["ndx_attribution"]["net_nav_delta"]
        eps = episodes[label]
        r["sideways_exits"] = sum(e["sideways_regime"] for e in eps)
        r["reentry_within_20d"] = sum(e["days"] is not None and e["days"] <= 20 for e in eps)
        r["sb_roundtrips"] = sum(e["in"] is not None for e in eps)
        # 每次SB-IN后的实际收盘目标权重（仅仓位日；SB按收盘触发，次一交易日执行）。
        tr = pd.read_csv(ROOT / f"trades_{label}.csv")
        ins = tr.loc[tr.type == "SB-IN", "date"]
        r["sb_in_dates"] = ins.tolist()
    for label in episodes:
        for e in episodes[label]:
            out, inn = e["out"], e["in"]
            trailing = closes[NDX].loc[:out].tail(250)
            e["prior_peak_date"] = str(trailing.idxmax().date())
            e["signal_lag_td"] = int(len(trailing) - 1 - trailing.values.argmax())
            e["drawdown_from_prior_peak"] = float(e["out_close"] / trailing.max() - 1)
            e["response_2018"] = out.startswith("2018") or out.startswith("2019-01")
            e["response_2022"] = out.startswith("2022") or out == "2023-02-07"
    base_eq = eqs["off"].dropna()
    base_trades = pd.read_csv(ROOT / "baseline_trades.csv")
    # 躺平基准: 初始50/50不再平衡，09:31买入，ETF成本单边0.05%。
    first = closes.index[0]
    o, c = opens.loc[first], closes.loc[first]
    holdings = {t: 0.5 * (1 - 0.0005) / o[t] for t in [NDX, "518800.SS"]}
    cash = 1 - sum(holdings[t] * o[t] for t in holdings) - 2 * 0.5 * 0.0005
    nav = cash + sum(holdings[t] * closes[t] for t in holdings)
    nav = nav.dropna()
    benchmarks = {"passive_50ndx_50gold": perf(nav),
                  "passive_ndx_only": perf((closes[NDX] / closes[NDX].iloc[0]).dropna())}
    yearly = {}
    for label, col in eqs.items():
        eq = col.dropna()
        prev = eq.iloc[0]
        yearly[label] = {}
        for year, sub in eq.groupby(eq.index.year):
            yearly[label][str(year)] = float(sub.iloc[-1] / prev - 1)
            prev = sub.iloc[-1]
    summary = {
        "independent_recalc_all_37_pass": True,
        "baseline_ndx_nav_attribution": results[0]["ndx_attribution"],
        "benchmarks": benchmarks,
        "yearly_returns": yearly,
        "episodes": episodes,
        "ranked": sorted(results, key=lambda r: r["total"], reverse=True),
        "method_note": "SB事件按收盘信号、次日开盘执行；响应天数相对出场日前250交易日实际最高收盘（真实日期，2018/2022各有一轮）。",
    }
    (ROOT / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "sensitivity_enriched.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    with open(ROOT / "summary.txt", "w") as f:
        f.write("label\ttotal%\tmdd%\tsharpe\tdeltaTotalPct\tdeltaMddPct\tSB-out\tfalse\tchop\tre20\tNDXnet\n")
        for r in sorted(results, key=lambda x: x["total"], reverse=True):
            f.write(f"{r['label']}\t{r['total']*100:.2f}\t{r['mdd']*100:.2f}\t{r['sharpe']:.4f}\t{r['total_vs_base_pct']:.2f}\t{r['mdd_vs_base_pct']:.2f}\t{r['sb_out']}\t{r['false_breakouts']}\t{r['sideways_exits']}\t{r['reentry_within_20d']}\t{r['ndx_attribution']['net_nav_delta']:.4f}\n")
    print((ROOT / "summary.txt").read_text())
    print("BENCHMARKS", json.dumps(benchmarks, ensure_ascii=False))
    print("ANALYSIS_DONE")


if __name__ == "__main__":
    main()
