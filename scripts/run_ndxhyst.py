#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIG-NDXHYST（2026-11-20，哥哥令「纳指要和黄金一致，回测一下～」/ yifei seq6105）

纳指对齐黄金失势滞回机制两臂对照：
  臂A = 正典原样（sb_days=0，纳指永持，run_v31_canonical 默认参数零改动）
  臂B = 纳指启用与黄金同构失势滞回：收盘<MA250×0.98 → NDX-OFF（sb_days=1, sb_exit_buf=0.98），
        收盘>MA250×1.02 → NDX-ON（sb_re_days=1, sb_re_buf=1.02）；±2% 滞回带，单日触发/单日回归。
        换防去向 sb_to="banks"（与黄金失势「换银行各行 MA250 过滤」最同构，派单口径第一选项）。
数据 = 当日冻结 y-data.csv（至 20261118）+ 既有 llm_monthly/llm_events（零新增 LLM 调用）。
口径与 20261031 SB 研究单差异披露：旧单 sb_to=cash（默认）+组合归一吸回；本单 sb_to=banks。
"""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/claude_big/scripts")
import rotation_backtest as rb

ROOT = Path("/tmp/ndxhyst")
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)
NDX = "513100.SS"
GOLD = "518800.SS"
BANKS = ["601398.SS", "601988.SS", "601939.SS", "601288.SS"]
COST = 0.0005


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
    """年换手 = Σ|Δw|(全资产含漂移校正近似: 直接相邻日权重差绝对值和)/2 按年折算。
    正典口径采用「每日实际权重绝对变动合计」的年度均值（含漂移），与既有报告可比性披露。"""
    dw = wdf.diff().abs().sum(axis=1)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float(dw.sum() / years)


def attribution_ndx(eq, wdf, closes):
    """纳指腿 NAV 贡献（上日权重×当日收益 - 换手费估计），与 20261031 单同式。"""
    r = closes.reindex(eq.index).pct_change().fillna(0.0)
    prev_w = wdf.reindex(eq.index).shift(1).fillna(0.0)
    gross = eq.shift(1) * prev_w[NDX] * r[NDX]
    port_r = eq.pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
    drift_w = prev_w[NDX] * (1 + r[NDX]) / (1 + port_r)
    turnover = (wdf[NDX].reindex(eq.index) - drift_w).abs()
    fee = eq * turnover * COST
    daily = gross - fee
    return {"gross_nav_delta": float(gross.sum()), "estimated_fee": float(fee.sum()),
            "net_nav_delta": float(daily.sum())}


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
        # 后续20交易日效果（OFF信号日起，含未回归窗截断）
        i0 = days.get_loc(d)
        i20 = min(i0 + 20, len(days) - 1)
        d20 = days[i20]
        ndx_ret20 = float(closes[NDX].loc[d20] / px - 1)
        b_ret20 = float(eq.loc[d20] / eq.loc[d] - 1)
        a_ret20 = float(base_eq.loc[d20] / base_eq.loc[d] - 1)
        res.append({
            "off_date": str(d.date()), "on_date": str(end.date()) if end is not None else None,
            "days_out": int(days.get_loc(end) - i0) if end is not None else None,
            "off_close": px, "off_vs_ma250": float(px / ma - 1),
            "on_close": float(closes.loc[end, NDX]) if end is not None else None,
            "ndx_ret_during_off": float(closes.loc[end, NDX] / px - 1) if end is not None else None,
            "min_ret_during_off": float(seg.min() / px - 1),
            "ndx_ret_20d": ndx_ret20,
            "armB_ret_20d": b_ret20, "armA_ret_20d": a_ret20,
            "dodge_20d": bool(ndx_ret20 < 0),  # 躲过续跌
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

    # ---- 臂A：正典原样 ----
    eqA, trA, wA = rb.bt_v31(closes, opens, **kw)
    # 正典锚断言：与 run_v31_canonical 全等（数据/文档同源即应逐位一致）
    eqC, trC, wC = rb.run_v31_canonical(closes, opens)
    pd.testing.assert_series_equal(eqA, eqC)
    pd.testing.assert_frame_equal(wA, wC)
    assert trA == trC
    assert not any(t[1].startswith("SB-") for t in trA)
    print("ANCHOR armA == run_v31_canonical: PASS", flush=True)

    # ---- 臂B：纳指黄金同构滞回 ----
    eqB, trB, wB = rb.bt_v31(closes, opens,
                             sb_days=1, sb_re_days=1,
                             sb_exit_buf=0.98, sb_re_buf=1.02,
                             sb_to="banks", **kw)
    typesB = Counter(t[1] for t in trB)
    print("armB SB-OUT:", typesB["SB-OUT"], "SB-IN:", typesB["SB-IN"], flush=True)

    days = closes.index
    ep = episodes(trB, closes, eqB, eqA, days)
    (OUT / "ndx_off_on_episodes.json").write_text(json.dumps(ep, ensure_ascii=False, indent=2))

    summary = {
        "asof": "2026-11-20",
        "data_sha256": sha256(data_path),
        "data_last": str(days[-1].date()), "days": len(days),
        "armA": {**perf(eqA), "trades": len(trA), "turnover_annual": turnover_annual(wA, eqA),
                 "ndx_attr": attribution_ndx(eqA, wA, closes)},
        "armB": {**perf(eqB), "trades": len(trB), "turnover_annual": turnover_annual(wB, eqB),
                 "ndx_attr": attribution_ndx(eqB, wB, closes),
                 "sb_out": int(typesB["SB-OUT"]), "sb_in": int(typesB["SB-IN"])},
        "yearly_A": {str(k): float(v) for k, v in yearly_pnl(eqA).items()},
        "yearly_B": {str(k): float(v) for k, v in yearly_pnl(eqB).items()},
        "yearly_delta": {str(k): float(v) for k, v in
                         (pd.Series(yearly_pnl(eqB)) - pd.Series(yearly_pnl(eqA))).dropna().items()},
        "episodes_n": len(ep),
        "ep_dodge_rate_20d": float(np.mean([e["dodge_20d"] for e in ep])) if ep else None,
        "ep_avg_off_days": float(np.mean([e["days_out"] for e in ep if e["days_out"] is not None])) if ep else None,
        "anchor_canonical_match": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # 落盘原始件
    for tag, eq, tr, w in (("armA", eqA, trA, wA), ("armB", eqB, trB, wB)):
        eq.to_csv(OUT / f"equity_{tag}.csv", header=["equity"])
        w.round(6).to_csv(OUT / f"weights_{tag}.csv")
        pd.DataFrame(tr, columns=["date", "type", "ticker", "note"]).to_csv(
            OUT / f"trades_{tag}.csv", index=False)
    print(json.dumps({k: summary[k] for k in ["episodes_n", "ep_dodge_rate_20d"]}), flush=True)
    print("A_total", summary["armA"]["total"], "B_total", summary["armB"]["total"], flush=True)
    print("A_mdd", summary["armA"]["mdd"], "B_mdd", summary["armB"]["mdd"], flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
