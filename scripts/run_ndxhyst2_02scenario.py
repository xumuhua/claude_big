#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIG-NDXHYST2 任务B（yifei seq6111）：02 剧本合成压力窗压测

口径：
- 窗 = 2000-03-01 → 2002-10-31（^IXIC 真实 -77.9% 崩盘路径），交易日历=IXIC 美股日历。
- 513100.SS 价格 = 合成序列：IXIC 日收益累乘，锚定窗首日 1.0（未上市 ETF 的等价替代）。
- 518800.SS 黄金价格 = 月度金价（datahub LBMA 月均 2000-03→2002-10）线性插值到日频
  + 2001-09→2002-10 段用 tushare fx_daily XAUUSD 日频覆盖（两段在 2001-09 缝合，
  收益序列连续）。日频收益是机制触发所需的最小粒度。
- 四银行 = 现金替代（收益恒 0，按派单口径「A股银行若无则用现金替代并标注」）。
- 其他资产（588000/159920/159952）= 窗内不存在（2000-02 未上市），填 NaN 不参选。
- LLM 月度/事件文档 = 禁用（use_llm_exit=False, use_events=False）——合成窗内
  复用 2013-2026 文档属穿越/污染；纯机制压测。
- 压测臂 = 臂A 正典 + 任务A 四档参数臂（同 sb 参数）。
- 只回答「02 年级重演时各臂死多惨」：逐日净值/MDD/回本时长；不与历史段读数混排。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/home/claude/claude_big/scripts")
import rotation_backtest as rb

ROOT = Path("/tmp/ndxhyst2")
OUT = ROOT / "out"
NDX = "513100.SS"
GOLD = "518800.SS"
BANKS = ["601398.SS", "601988.SS", "601939.SS", "601288.SS"]

ARMS = [("armA", None, None)] + [
    ("b98", 0.98, 1.02), ("b95", 0.95, 1.05), ("b90", 0.90, 1.10), ("b85", 0.85, 1.15),
]


def build_synth():
    """构造合成 closes/opens 矩阵（IXIC 日历）。"""
    ix = pd.read_csv(ROOT / "data/ixic_2000_2002.csv")
    ix["date"] = pd.to_datetime(ix["trade_date"], format="%Y%m%d")
    ix = ix.set_index("date").sort_index()
    ixic_ret = ix["close"].pct_change().fillna(0.0)

    gm = pd.read_csv(ROOT / "data/gold_monthly_1833_2026.csv")
    gm["date"] = pd.to_datetime(gm["Date"])
    gm = gm.set_index("date").sort_index()
    # 切片前移一个月（首锚+14日≈2000-02-15 < 窗首 3-01，插值管道首段不留空）
    gm = gm.loc["2000-01-05":"2002-11-30", "Price"]

    xau = pd.read_csv(ROOT / "data/xau_daily_200109_200210.csv")
    xau["date"] = pd.to_datetime(xau["trade_date"], format="%Y%m%d")
    xau = xau.set_index("date").sort_index()

    days = ix.index
    # --- 黄金日价：统一「收益映射到 IXIC 日历」管道 ---
    # 周日历(XAU)/月历(LBMA)都先在自身日历上算收益，reindex 到 IXIC 日历（缺失日收益=0，
    # 即无新信息持有），再从窗首锚价累乘——避免不同日历的行直接对位（KeyError 家族坑）。
    monthly_anchors = gm.copy()
    monthly_anchors.index = monthly_anchors.index + pd.offsets.Day(14)  # 月均≈月中
    anchors = monthly_anchors.reindex(monthly_anchors.index.union(days)).interpolate("time").reindex(days)
    anchors = anchors.bfill().ffill()  # 保险：首尾无锚段用边界价延拓（时间插值不留 NaN）
    interp_ret = anchors.pct_change().fillna(0.0)
    gold_px_interp = float(anchors.iloc[0]) * (1 + interp_ret).cumprod()  # 月度插值全程段

    xau_ret_own = xau["bid_close"].pct_change().fillna(0.0)  # XAU 自身周日历收益
    first_xau_day = xau.index[0]
    # XAU 段：以插值管道在 first_xau_day 的累计价续锚，收益映射到 IXIC 日历
    xau_ret_on_ixic = xau_ret_own.reindex(days).fillna(0.0)
    xau_ret_on_ixic.loc[days[0]] = 0.0  # 首日无前收，置 0
    # XAU 首日是周日（2001-09-09）不在 IXIC 日历——锚价取插值管道 asof（≤该日的最后 IXIC 日）
    anchor_series = gold_px_interp.loc[gold_px_interp.index <= first_xau_day]
    anchor_px = float(anchor_series.iloc[-1])
    synth_tail = anchor_px * (1 + xau_ret_on_ixic).cumprod()

    gold_daily = gold_px_interp.copy()
    mask = days >= first_xau_day
    gold_daily.loc[mask] = synth_tail.loc[mask]
    # 缝合检验：换轨日前最后一个插值日 vs 首个 XAU 映射日的收益跳变（<1% 视为可接受）
    before = gold_daily.loc[~mask].iloc[-1] if (~mask).any() else float(gold_daily.iloc[0])
    first_mapped = gold_daily.loc[mask].iloc[0]
    stitch_jump = float(first_mapped / before - 1)

    # --- 合成价格矩阵 ---
    closes = pd.DataFrame(index=days)
    closes[NDX] = (1 + ixic_ret).cumprod()           # 锚 1.0
    closes[GOLD] = gold_daily / gold_daily.iloc[0]   # 锚 1.0
    for b in BANKS:
        closes[b] = 1.0                              # 现金替代（收益0，派单口径标注）
    # B_TRACK 三标的窗内未上市：全 NaN 列（引擎 notna 逻辑天然跳过，勿删列）
    for nb in ["588000.SS", "159920.SZ", "159952.SZ"]:
        closes[nb] = np.nan
    opens = closes.copy()                            # 开盘价=收盘价（合成窗无日内结构，披露）
    return closes, opens, {
        "ixic_total_ret": float(ix["close"].iloc[-1] / ix["close"].iloc[0] - 1),
        "ixic_peak": float(ix["close"].max()), "ixic_trough": float(ix["close"].min()),
        "gold_window_ret": float(gold_daily.iloc[-1] / gold_daily.iloc[0] - 1),
        "gold_stitch_jump_200109": stitch_jump,
        "days": len(days), "first": str(days[0].date()), "last": str(days[-1].date()),
    }


def recovery_days(eq):
    """从 MDD 谷底回到前高所需交易日数（未回本=None）。"""
    peak_idx = eq.idxmax()
    cm = eq.cummax()
    trough = (eq / cm - 1).idxmin()
    pre_peak = eq.loc[:trough].cummax().iloc[-1]
    after = eq.loc[trough:]
    rec = after[after >= pre_peak]
    if len(rec) == 0:
        return None, str(trough.date())
    return int(eq.index.get_loc(rec.index[0]) - eq.index.get_loc(trough)), str(trough.date())


def main():
    closes, opens, meta = build_synth()
    print("synth meta:", json.dumps(meta, ensure_ascii=False), flush=True)
    closes.to_csv(OUT / "synth_closes_02.csv")
    opens.to_csv(OUT / "synth_opens_02.csv")

    kw = dict(use_llm_exit=False, use_events=False)  # 合成窗禁用 LLM 文档
    results = {}
    for tag, xb, rbv in ARMS:
        if xb is None:
            eq, tr, w = rb.bt_v31(closes, opens, **kw)
        else:
            eq, tr, w = rb.bt_v31(closes, opens,
                                  sb_days=1, sb_re_days=1,
                                  sb_exit_buf=xb, sb_re_buf=rbv,
                                  sb_to="banks", **kw)
        eq.to_csv(OUT / f"synth_equity_{tag}.csv", header=["equity"])
        w.round(6).to_csv(OUT / f"synth_weights_{tag}.csv")
        pd.DataFrame(tr, columns=["date", "type", "ticker", "note"]).to_csv(
            OUT / f"synth_trades_{tag}.csv", index=False)
        mdd = float((eq / eq.cummax() - 1).min())
        rec_days, trough_d = recovery_days(eq)
        sb_outs = [t for t in tr if t[1] == "SB-OUT"]
        results[tag] = {
            "total": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "mdd": mdd, "trough_date": trough_d,
            "recovery_days_from_trough": rec_days,
            "sb_out_n": len(sb_outs),
            "sb_out_dates": [t[0] for t in sb_outs],
            "end": float(eq.iloc[-1]),
        }
        print(f"{tag}: total={results[tag]['total']:+.4f} mdd={mdd:.4f} trough={trough_d} "
              f"rec={rec_days} sb_out={len(sb_outs)} {[t[0] for t in sb_outs]}", flush=True)

    (OUT / "summary_B.json").write_text(json.dumps(
        {"meta": meta, "arms": results}, ensure_ascii=False, indent=2))
    print("ALL_DONE_B", flush=True)


if __name__ == "__main__":
    main()
