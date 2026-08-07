#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""y-data.csv 全量重建器 (claude_big 实盘数据底座)

数据源: ~/local_data/common_data/big_pool_price.csv (本机=数据权威)
  由 quant_levels/fetch_stock_price.py::fetch_big_pool() 每日生成:
  银行股=common_data前复权文件, ETF/LOF=fund_daily原始价×fund_adj复权因子(前复权)

为什么全量重建而非增量追加 (20260807 复权口径切换):
  前复权锚随新分红/拆分事件漂移, 只有全量重写才能保证全序列同一口径。
  原始价时代的结构性假跳变必须消除: 518800 2026-04-15 份额合并(raw假+6.7%)、
  513100 2022-01-14 拆分(raw假-80%, 引擎|ret|>30%修复降级为冗余保险)、
  银行股历年除息假跌-5~-6%(raw口径白白丢了股息收益)。

文件格式 (与历史一致):
  - 倒序存储 (新日期在前), 8 bar/日: 1500..0931
  - 0931 bar=当日开盘价(复权), 1500 bar=收盘价(复权), 中间6 bar=收盘占位(无消费者)
  - 上市前=-1.0 (引擎转NaN), 停牌/缺口=前收盘顺延
  - 辅助列 (_fact/_ln/_base/_avgs32/_avgs128) 无任何消费者 (全仓grep验证),
    重建时按新口径直接重算: _fact=1, _base=首个有效收盘, _ln=ln(px)+const
    (const=1-min(ln(px)), 使min(_ln)=1), _avgs=逐bar EMA(alpha=1/32,1/128)

校验 (--check 或每次运行自动执行):
  新旧文件重叠日【日收益】逐票比对 —— 复权口径下仅分红/拆分事件的临近日收益
  允许不同(那才是修复点), 其余日期偏差>0.05%即报警(不中止, 人工判读)。

用法:
  python3 scripts/update_ydata.py            # 全量重建 y-data.csv
  python3 scripts/update_ydata.py --check    # 只校验不写入
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YDATA = os.path.join(ROOT, "y-data.csv")
LOCAL = os.path.expanduser("~/local_data/common_data/big_pool_price.csv")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotation_backtest import TICKERS  # noqa: E402

BAR_TIMES = ["0931", "1000", "1031", "1100", "1331", "1400", "1431", "1500"]


def build_frames(local_df):
    """big_pool_price.csv → 每票 (calendar对齐的 open/close 序列), 返回交易日历。"""
    local_df = local_df.copy()
    local_df["date"] = pd.to_datetime(local_df.trade_date, format="%Y%m%d")
    cal = sorted(local_df.date.unique())
    series = {}
    for full in TICKERS:
        sub = local_df[local_df.code == full].set_index("date")[["open", "close"]]
        sub = sub.astype(float).sort_index()
        series[full] = sub
    return pd.DatetimeIndex(cal), series


def rebuild(verbose=True):
    """全量重建 y-data.csv。返回 (新旧收益偏差报告 DataFrame)。"""
    if not os.path.exists(LOCAL):
        raise RuntimeError(f"本地数据源缺失: {LOCAL} (先跑 fetch_stock_price.py)")
    local = pd.read_csv(LOCAL, dtype={"trade_date": str})
    cal, series = build_frames(local)
    if verbose:
        print(f"本地数据: {cal[0].date()} -> {cal[-1].date()}, {len(cal)} 个交易日")

    # 1) 每票对齐到全日历: 上市前 -1.0, 上市后缺口 ffill 顺延
    aligned = {}
    for full, s in series.items():
        o = s["open"].reindex(cal)
        c = s["close"].reindex(cal)
        first_valid = c.first_valid_index()
        c = c.ffill()
        o = o.ffill()
        pre = cal < first_valid if first_valid is not None else np.ones(len(cal), bool)
        c[pre] = -1.0
        o[pre] = -1.0
        aligned[full] = (o.values, c.values)
        if verbose:
            print(f"  {full} {TICKERS[full]}: 首个有效日 "
                  f"{first_valid.date() if first_valid is not None else '无'}")

    # 2) 辅助列参数 (新口径重算; 无消费者, 仅保持schema)
    aux = {}
    for full, (o_, c_) in aligned.items():
        valid = c_ > 0
        lnv = np.log(c_[valid])
        aux[full] = {"const": float(1 - lnv.min()) if valid.any() else 0.0,
                     "base": float(c_[valid][0]) if valid.any() else 1.0,
                     "a32": 0.0, "a128": 0.0, "init": False}

    # 3) 逐日构造 8 bar (EMA按时间序推进, 存储倒序)
    rows = []
    for i, d in enumerate(cal):
        for hm in BAR_TIMES:     # EMA 时间序 0931->1500
            for full in TICKERS:
                o_, c_ = aligned[full]
                px = o_[i] if hm == "0931" else c_[i]
                if px <= 0:
                    continue
                ln = np.log(px) + aux[full]["const"]
                if not aux[full]["init"]:
                    aux[full]["a32"] = aux[full]["a128"] = ln
                    aux[full]["init"] = True
                else:
                    aux[full]["a32"] += (ln - aux[full]["a32"]) / 32
                    aux[full]["a128"] += (ln - aux[full]["a128"]) / 128
        # 当日 avgs 快照(8 bar 后状态)供所有行占位
        snap = {f: (aux[f]["a32"], aux[f]["a128"]) for f in TICKERS}
        for hm in reversed(BAR_TIMES):
            row = {"trade_date": d.strftime("%Y%m%d") + hm}
            for full in TICKERS:
                o_, c_ = aligned[full]
                px = o_[i] if hm == "0931" else c_[i]
                row[full] = px
                row[full + "_fact"] = 1
                row[full + "_ln"] = (np.log(px) + aux[full]["const"]) if px > 0 else -1.0
                row[full + "_base"] = aux[full]["base"]
                row[full + "_avgs128"] = snap[full][1]
                row[full + "_avgs32"] = snap[full][0]
            rows.append(row)
    cols = ["trade_date"] + [f + s for f in TICKERS
                             for s in ("", "_fact", "_ln", "_base", "_avgs128", "_avgs32")]
    new_df = pd.DataFrame(rows)[cols]
    return new_df


def diff_report(old_df, new_df):
    """新旧重叠日 日收益 逐票比对 (1500 bar)。偏差>0.05%的日期列清单。"""
    def closes_of(df):
        d = df[df.trade_date.str.endswith("1500")].copy()
        d["date"] = pd.to_datetime(d.trade_date.str[:8], format="%Y%m%d")
        return d.set_index("date").sort_index()
    old_c, new_c = closes_of(old_df), closes_of(new_df)
    common = old_c.index.intersection(new_c.index)
    reports = {}
    for full in TICKERS:
        o = pd.to_numeric(old_c.loc[common, full], errors="coerce")
        n = pd.to_numeric(new_c.loc[common, full], errors="coerce")
        ro = o.pct_change()
        rn = n.pct_change()
        dev = (rn - ro).abs()
        bad = dev[dev > 0.0005].dropna()
        reports[full] = bad
    return reports


def update(check_only=False, verbose=True):
    old_df = None
    if os.path.exists(YDATA):
        old_df = pd.read_csv(YDATA, dtype={"trade_date": str})
    new_df = rebuild(verbose=verbose)

    if old_df is not None:
        reports = diff_report(old_df, new_df)
        total_bad = sum(len(r) for r in reports.values())
        print(f"\n新旧日收益比对 (偏差>0.05% 的日期数/票):")
        for full, bad in reports.items():
            if len(bad):
                ds = [str(d.date()) for d in bad.index]
                show = ds if len(ds) <= 8 else ds[:8] + [f"...共{len(ds)}日"]
                print(f"  {full} {TICKERS[full]}: {len(bad)} 日 {show}")
        print(f"  合计 {total_bad} 日 (应≈分红/拆分事件日: 银行每年1-2次+513100拆分+518800合并)")
    if check_only:
        print("--check 模式: 未写入")
        return []
    new_df.to_csv(YDATA, index=False)
    if verbose:
        print(f"\n已写入 {YDATA}: {len(new_df)} 行 "
              f"({new_df.trade_date.str[:8].iloc[-1]} -> {new_df.trade_date.str[:8].iloc[0]})")
    return new_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验不写入")
    args = ap.parse_args()
    update(check_only=args.check)
