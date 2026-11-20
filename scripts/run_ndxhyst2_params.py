#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIG-NDXHYST2 任务A（2026-11-20，哥哥令「不接受，增补到和黄金一样」→方案②参数族寻优 / yifei seq6111）

纳指崩盘保险丝·滞回带参数族回测（历史段 2013-08-01→20261118）：
  臂A    = 正典原样（sb_days=0 永持，锚断言≡run_v31_canonical，引用 NDXHYST 已锚读数）
  臂矩阵 = sb_exit_buf ∈ {0.98, 0.95, 0.90, 0.85}，sb_re_buf=镜像对称(2-exit_buf)，
           sb_days=1/sb_re_days=1，sb_to="banks"（与 NDXHYST 同构，零码件参数化）
数据 = 当日冻结 y-data.csv（至 20261118）+ 既有 llm_monthly/llm_events（零新增 LLM）。
口径披露：0.98 臂与 NDXHYST 臂B 同参数（引用关系，本单重跑复核）。
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/claude_big/scripts")
import rotation_backtest as rb

ROOT = Path("/tmp/ndxhyst2")
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)
NDX = "513100.SS"
COST = 0.0005

ARMS = [
    ("b98", 0.98, 1.02),
    ("b95", 0.95, 1.05),
    ("b90", 0.90, 1.10),
    ("b85", 0.85, 1.15),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def perf(eq):
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "first": str(eq.index[0].date()), "last": str(eq.index[-1].date()),
        "end": float(eq.iloc[-1]), "total": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "annual": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1),
        "mdd": float((eq / eq.cummax() - 1).min()),
        "sharpe": float(r.mean() / r.std() * np.sqrt(244)),
    }


def turnover_annual(wdf, eq):
    dw = wdf.diff().abs().sum(axis=1)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float(dw.sum() / years)


def episodes(trades, closes, eq, base_eq, days):
    outs = [t for t in trades if t[1] == "SB-OUT"]
    ins = [t for t in trades if t[1] == "SB-IN"]
    res = []
    for t in outs:
        d = pd.Timestamp(t[0])
        nxt = next((x for x in ins if pd.Timestamp(x[0]) > d), None)
        end = pd.Timestamp(nxt[0]) if nxt else None
        seg = closes.loc[d:end, NDX] if end is not None else closes.loc[d:, NDX]
        px = float(closes.loc[d, NDX])
        ma = closes[NDX].rolling(250).mean().loc[d]
        i0 = days.get_loc(d)
        i20 = min(i0 + 20, len(days) - 1)
        d20 = days[i20]
        ndx_ret20 = float(closes[NDX].loc[d20] / px - 1)
        res.append({
            "off_date": str(d.date()), "on_date": str(end.date()) if end is not None else None,
            "days_out": int(days.get_loc(end) - i0) if end is not None else None,
            "off_close": px, "off_vs_ma250": float(px / ma - 1),
            "on_close": float(closes.loc[end, NDX]) if end is not None else None,
            "ndx_ret_during_off": float(closes.loc[end, NDX] / px - 1) if end is not None else None,
            "min_ret_during_off": float(seg.min() / px - 1),
            "ndx_ret_20d": ndx_ret20,
            "armX_ret_20d": float(eq.loc[d20] / eq.loc[d] - 1),
            "armA_ret_20d": float(base_eq.loc[d20] / base_eq.loc[d] - 1),
            "dodge_20d": bool(ndx_ret20 < 0),
            "eq_ret_out_to_in": float(eq.loc[end] / eq.loc[d] - 1) if end is not None else None,
            "base_eq_ret_same_window": float(base_eq.loc[end] / base_eq.loc[d] - 1) if end is not None else None,
        })
    return res


def yearly_pnl(eq):
    y = eq.resample("YE").last()
    y.index = y.index.year
    prev = pd.Series([eq.iloc[0]], index=[eq.index[0].year - 1])
    full = pd.concat([prev, y])
    return (full.pct_change().dropna()).to_dict()


def main():
    data_path = Path("/home/claude/claude_big/y-data.csv")
    rb.DATA = str(data_path)
    closes, opens = rb.load_daily()
    closes, opens = closes.loc["2013-08-01":], opens.loc["2013-08-01":]
    kw = dict(llm_dir="/home/claude/claude_big/output/llm_monthly",
              events_dir="/home/claude/claude_big/output/llm_events")

    # ---- 臂A：正典原样（锚断言复用 NDXHYST 口径） ----
    eqA, trA, wA = rb.bt_v31(closes, opens, **kw)
    eqC, trC, wC = rb.run_v31_canonical(closes, opens)
    pd.testing.assert_series_equal(eqA, eqC)
    pd.testing.assert_frame_equal(wA, wC)
    assert trA == trC
    print("ANCHOR armA == run_v31_canonical: PASS", flush=True)

    days = closes.index
    summary = {
        "asof": "2026-11-20", "data_sha256": sha256(data_path),
        "data_last": str(days[-1].date()), "days": len(days),
        "armA": {**perf(eqA), "trades": len(trA), "turnover_annual": turnover_annual(wA, eqA)},
        "arms": {},
    }

    for tag, xb, rbv in ARMS:
        eqX, trX, wX = rb.bt_v31(closes, opens,
                                 sb_days=1, sb_re_days=1,
                                 sb_exit_buf=xb, sb_re_buf=rbv,
                                 sb_to="banks", **kw)
        types = Counter(t[1] for t in trX)
        ep = episodes(trX, closes, eqX, eqA, days)
        (OUT / f"episodes_{tag}.json").write_text(json.dumps(ep, ensure_ascii=False, indent=2))
        eqX.to_csv(OUT / f"equity_{tag}.csv", header=["equity"])
        wX.round(6).to_csv(OUT / f"weights_{tag}.csv")
        pd.DataFrame(trX, columns=["date", "type", "ticker", "note"]).to_csv(
            OUT / f"trades_{tag}.csv", index=False)
        yd = {str(k): float(v) for k, v in
              (pd.Series(yearly_pnl(eqX)) - pd.Series(yearly_pnl(eqA))).dropna().items()}
        summary["arms"][tag] = {
            "sb_exit_buf": xb, "sb_re_buf": rbv,
            **perf(eqX), "trades": len(trX), "turnover_annual": turnover_annual(wX, eqX),
            "sb_out": int(types["SB-OUT"]), "sb_in": int(types["SB-IN"]),
            "delta_total": float(eqX.iloc[-1] / eqX.iloc[0] - 1 - (eqA.iloc[-1] / eqA.iloc[0] - 1)),
            "delta_mdd": float((eqX / eqX.cummax() - 1).min() - (eqA / eqA.cummax() - 1).min()),
            "delta_sharpe": float(pd.Series(eqX.pct_change().dropna()).mean() / eqX.pct_change().dropna().std() * np.sqrt(244)
                                  - pd.Series(eqA.pct_change().dropna()).mean() / eqA.pct_change().dropna().std() * np.sqrt(244)),
            "episodes_n": len(ep),
            "ep_dodge_rate_20d": float(np.mean([e["dodge_20d"] for e in ep])) if ep else None,
            "ep_avg_off_days": float(np.mean([e["days_out"] for e in ep if e["days_out"] is not None])) if any(e["days_out"] is not None for e in ep) else None,
            "yearly_delta": yd,
        }
        print(f"{tag} exit={xb} re={rbv}: total={summary['arms'][tag]['total']:.4f} "
              f"(Δ{summary['arms'][tag]['delta_total']:+.4f}) mdd={summary['arms'][tag]['mdd']:.4f} "
              f"out={types['SB-OUT']} dodge={summary['arms'][tag]['ep_dodge_rate_20d']}", flush=True)

    # 臂A 原始件也落（对照用）
    eqA.to_csv(OUT / "equity_armA.csv", header=["equity"])
    wA.round(6).to_csv(OUT / "weights_armA.csv")
    pd.DataFrame(trA, columns=["date", "type", "ticker", "note"]).to_csv(
        OUT / "trades_armA.csv", index=False)
    summary["yearly_A"] = {str(k): float(v) for k, v in yearly_pnl(eqA).items()}

    (OUT / "summary_A.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("A_total", summary["armA"]["total"], "A_mdd", summary["armA"]["mdd"], flush=True)
    print("ALL_DONE_A", flush=True)


if __name__ == "__main__":
    main()
