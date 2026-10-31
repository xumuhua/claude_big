#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纳指腿慢性破位滞回研究（2026-10-31，哥哥令 / yifei seq4456）。

离线正典引擎 + 当日冻结数据/LLM文档；只研究，不改实盘配置。
执行: systemd-run --user --unit=sb-hysteresis-20261031 ...
"""
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import rotation_backtest as rb

ROOT = Path("/data/workspace/quant/sb_hysteresis_20261031")
NDX = "513100.SS"
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


def period_perf(eq, start, end):
    x = eq.loc[start:end]
    return perf(x) if len(x) > 1 else {}


def attribution(eq, wdf, closes):
    """按上日收盘实际持仓权重拆NAV贡献，扣除NDX换手费估计；非独立资产收益。"""
    r = closes.reindex(eq.index).pct_change().fillna(0.0)
    prev_w = wdf.reindex(eq.index).shift(1).fillna(0.0)
    gross = eq.shift(1) * prev_w[NDX] * r[NDX]
    # 上日权重按当日收益漂移至开盘前，和本日权重之差近似执行换手。
    port_r = eq.pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
    drift_w = prev_w[NDX] * (1 + r[NDX]) / (1 + port_r)
    turnover = (wdf[NDX].reindex(eq.index) - drift_w).abs()
    fee = eq * turnover * COST
    daily = gross - fee
    return {
        "gross_nav_delta": float(gross.sum()), "estimated_fee": float(fee.sum()),
        "net_nav_delta": float(daily.sum()),
        "all_assets_return_sum_check": float((prev_w[[c for c in wdf.columns if c != "cash"]] * r).sum(axis=1).sum()),
    }


def sb_events(trades, closes, eq, base_eq):
    ts = [dict(zip(["date", "type", "name", "note"], t)) for t in trades]
    for t in ts:
        d = pd.Timestamp(t["date"])
        t["ndx_close"] = float(closes.loc[d, NDX])
        t["equity"] = float(eq.loc[d])
    return ts


def episodes(trades, closes, eq, base_eq, days):
    out = [t for t in trades if t[1] == "SB-OUT"]
    ins = [t for t in trades if t[1] == "SB-IN"]
    result = []
    for i, t in enumerate(out):
        d = pd.Timestamp(t[0])
        entry = next((x for x in ins if pd.Timestamp(x[0]) > d), None)
        end = pd.Timestamp(entry[0]) if entry else None
        segment = closes.loc[d:end, NDX] if end is not None else closes.loc[d:, NDX]
        ma = closes[NDX].rolling(250).mean().loc[d]
        r120 = closes[NDX].loc[:d].iloc[-1] / closes[NDX].loc[:d].iloc[-121] - 1 if len(closes[NDX].loc[:d]) > 120 else np.nan
        px = float(closes.loc[d, NDX])
        duration = int(days.get_loc(end) - days.get_loc(d)) if end is not None else None
        is_regime = bool(np.isfinite(r120) and abs(px / ma - 1) <= 0.08 and abs(r120) <= 0.10)
        # 明示研究口径：震荡区退出、60交易日内回归且途中未再跌10%。
        false_break = bool(is_regime and end is not None and duration <= 60
                           and segment.min() / px - 1 > -0.10)
        result.append({
            "out": str(d.date()), "in": str(end.date()) if end is not None else None,
            "days": duration, "out_close": px,
            "in_close": float(closes.loc[end, NDX]) if end is not None else None,
            "price_return_to_in": float(closes.loc[end, NDX] / px - 1) if end is not None else None,
            "min_return": float(segment.min() / px - 1), "max_return": float(segment.max() / px - 1),
            "exit_vs_ma250": float(px / ma - 1), "ret120": float(r120),
            "sideways_regime": is_regime, "false_breakout_research_def": false_break,
            "eq_return_out_to_in": float(eq.loc[end] / eq.loc[d] - 1) if end is not None else None,
            "base_return_same_window": float(base_eq.loc[end] / base_eq.loc[d] - 1) if end is not None else None,
        })
    return result


def main():
    rb.DATA = str(ROOT / "y-data.csv")
    closes, opens = rb.load_daily()
    closes, opens = closes.loc["2013-08-01":], opens.loc["2013-08-01":]
    kwargs = dict(llm_dir=str(ROOT / "output/llm_monthly"), events_dir=str(ROOT / "output/llm_events"))
    base_eq, base_trades, base_w = rb.bt_v31(closes, opens, **kwargs)
    canon_eq, canon_trades, canon_w = rb.run_v31_canonical(closes, opens)
    pd.testing.assert_series_equal(base_eq, canon_eq)
    pd.testing.assert_frame_equal(base_w, canon_w)
    assert base_trades == canon_trades
    assert not any(t[1].startswith("SB-") for t in base_trades)
    # 默认新参数须保持原实现（只有双计数分离，默认下语义完全一致）。
    legacy_eq, legacy_trades, legacy_w = rb.bt_v31(
        closes, opens, sb_days=5, sb_re_days=5, sb_exit_buf=0.98, sb_re_buf=1.0, **kwargs)
    assert len(legacy_trades) > len(base_trades)
    manifest = {
        "asof": "2026-10-31", "frozen_data_sha256": sha256(ROOT / "y-data.csv"),
        "data_rows": int(sum(1 for _ in open(ROOT / "y-data.csv")) - 1),
        "days": len(closes), "docs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "output").glob("llm_*/*.json"))
        },
        "canonical_baseline": perf(base_eq),
        "canonical_trades": len(base_trades),
        "canonical_match": True,
        "engine_note": "只加可选sb_exit_buf/sb_re_buf；默认正典sb_days=0未改。",
    }
    (ROOT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    pd.DataFrame(base_trades, columns=["date", "type", "ticker", "note"]).to_csv(ROOT / "baseline_trades.csv", index=False)
    base_eq.to_csv(ROOT / "baseline_equity.csv", header=["equity"])

    results, all_events, all_episodes = [], {}, {}
    saved = {}
    combos = [(0.98, 1.0, 0, 5)] + list(itertools.product([0.97, 0.98], [1.0, 1.02], [3, 5, 10], [3, 5, 10]))
    for exit_buf, re_buf, n, m in combos:
        label = f"off" if n == 0 else f"e{exit_buf:.2f}_r{re_buf:.2f}_n{n}_m{m}"
        if n == 0:
            eq, trades, wdf = base_eq, base_trades, base_w
        else:
            eq, trades, wdf = rb.bt_v31(closes, opens, sb_days=n, sb_re_days=m,
                                         sb_exit_buf=exit_buf, sb_re_buf=re_buf, **kwargs)
        ev = sb_events(trades, closes, eq, base_eq)
        ep = episodes(trades, closes, eq, base_eq, closes.index)
        types = Counter(t[1] for t in trades)
        row = {
            "label": label, "exit_buf": exit_buf, "re_buf": re_buf, "sb_days": n, "sb_re_days": m,
            **perf(eq), "trade_actions": len(trades), "sb_out": types["SB-OUT"], "sb_in": types["SB-IN"],
            "ndx_trade_actions": sum(t[2] == rb.TICKERS[NDX] for t in trades),
            "false_breakouts": sum(e["false_breakout_research_def"] for e in ep),
            "false_breakout_rate": float(np.mean([e["false_breakout_research_def"] for e in ep])) if ep else None,
            "completed_sb_days_median": float(np.median([e["days"] for e in ep if e["days"] is not None])) if ep else None,
            "ndx_attribution": attribution(eq, wdf, closes),
            "y2018": period_perf(eq, "2018-01-01", "2018-12-31"),
            "y2022": period_perf(eq, "2022-01-01", "2022-12-31"),
            "out_to_2018_end": period_perf(eq, "2018-01-01", "2019-01-31"),
            "out_to_2022_end": period_perf(eq, "2022-01-01", "2023-02-28"),
        }
        results.append(row)
        all_events[label] = ev
        all_episodes[label] = ep
        # 每一档留净值与交易明细；36档CSV很小，保证报告数字可复算。
        pd.DataFrame(trades, columns=["date", "type", "ticker", "note"]).to_csv(ROOT / f"trades_{label}.csv", index=False)
        saved[label] = (eq, trades, wdf)
        print(json.dumps({k: row[k] for k in ["label", "total", "mdd", "sharpe", "trade_actions", "sb_out", "false_breakouts"]}), flush=True)

    (ROOT / "sensitivity.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (ROOT / "events.json").write_text(json.dumps(all_events, ensure_ascii=False, indent=2))
    (ROOT / "episodes.json").write_text(json.dumps(all_episodes, ensure_ascii=False, indent=2))
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in results]).to_csv(ROOT / "sensitivity.csv", index=False)
    # 保存逐档净值，以便独立重算年度、熊市响应和贡献。
    pd.concat({k: v[0] for k, v in saved.items()}, axis=1).to_csv(ROOT / "all_equity.csv")
    base_attr = attribution(base_eq, base_w, closes)
    print("BASE_ATTRIBUTION", json.dumps(base_attr), flush=True)
    print("ALL_DONE", len(results), flush=True)


if __name__ == "__main__":
    main()
