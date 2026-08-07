#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""y-data.csv 每日增量更新器 (claude_big 实盘数据底座)

数据源 (20260807 起双源):
  主源 = tushare (fund_daily/daily 未复权, 用户指定接口, SH后缀),
         由 quant_levels/fetch_stock_price.py 每日抓取落盘
         ~/local_data/common_data/big_pool_price.csv (本机=数据权威)
  兜底 = 新浪 (akshare fund_etf_hist_sina / stock_zh_a_daily, 未复权原始价)
         —— 东财源(fund_etf_hist_em)限流严重不可用
  两源与 y-data.csv 主列口径均已验证一致 (2026-08-04 全池收盘价逐票比对零偏差;
  y-data 主列=未复权原始价, 拆分由引擎 |日收益|>30% 自动修复)

文件格式 (与历史一致):
  - 倒序存储 (新日期在前), 8 bar/日: 1500..0931
  - 0931 bar 填当日真实开盘价 (历史为9:31时点价, 语义同为"T+1开盘执行"口径)
  - 1500 bar 填收盘价; 中间6 bar(1000/1031/1100/1331/1400/1431)无任何消费者,
    填收盘价占位保持行数结构
  - 辅助列 (_fact/_ln/_base/_avgs32/_avgs128) 无任何消费者, 按原语义尽力延续:
    _fact=1, _base=末值, _ln=ln(price)+const(const从末行反解, 每票固定),
    _avgs=按时间序逐bar EMA(alpha=1/32,1/128)从末值继续
  - 停牌/缺数据日: 前一收盘顺延 (引擎对<=0转NaN, 顺延价等价于qfq的平直处理)

幂等: 已是最新则无新增, 可重复运行。
用法:
  python3 scripts/update_ydata.py            # 增量更新 y-data.csv
  python3 scripts/update_ydata.py --check    # 只检查不写入
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YDATA = os.path.join(ROOT, "y-data.csv")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotation_backtest import TICKERS, STOCKS  # noqa: E402

BAR_TIMES = ["0931", "1000", "1031", "1100", "1331", "1400", "1431", "1500"]
RETRY = 4


def _sina_symbol(code6):
    return ("sh" if code6[:2] in ("50", "51", "58", "60") else "sz") + code6


def fetch_daily(full_code, retries=RETRY):
    """新浪日线 (date/open/close), ETF/LOF 与个股两个接口。失败重试, 最终 None。"""
    import akshare as ak
    code6 = full_code[:6]
    sym = _sina_symbol(code6)
    for k in range(retries):
        try:
            if full_code in STOCKS:
                df = ak.stock_zh_a_daily(symbol=sym, adjust="")
            else:
                df = ak.fund_etf_hist_sina(symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "open", "close"]].dropna().set_index("date").astype(float)
        except Exception as e:
            print(f"  fetch {full_code} 第{k + 1}次失败: {str(e)[:80]}")
            time.sleep(2 * (k + 1))
    return None


def update(check_only=False, verbose=True):
    """增量更新 y-data.csv。返回新增交易日列表 (pd.Timestamp)。"""
    df = pd.read_csv(YDATA, dtype={"trade_date": str})
    df["dt"] = pd.to_datetime(df.trade_date, format="mixed")
    last_day = df.dt.dt.normalize().max()
    if verbose:
        print(f"y-data 当前截至 {last_day.date()}, {len(df)} 行")

    # 1) 先定新增交易日 (tushare本地文件的参考票, 无新增则直接退出, 零外网请求)
    LOCAL = os.path.expanduser("~/local_data/common_data/big_pool_price.csv")
    local = None
    if os.path.exists(LOCAL):
        local = pd.read_csv(LOCAL, dtype={"trade_date": str})
        local["date"] = pd.to_datetime(local.trade_date, format="%Y%m%d")

    def _from_local(full):
        if local is None:
            return None
        sub = local[local.code == full]
        if len(sub) and sub.date.max() > last_day:
            return sub.set_index("date")[["open", "close"]].astype(float)
        return None

    ref = _from_local("513100.SS")
    if ref is None:
        ref = fetch_daily("513100.SS")     # 本地无新增时才问新浪(判定是否有新交易日)
        if ref is None:
            raise RuntimeError("数据抓取失败: 513100.SS (双源均失败, 中止保安全)")
    new_days = sorted(d for d in ref.index if d > last_day)
    if not new_days:
        if verbose:
            print("已是最新, 无新增交易日")
        return []
    if verbose:
        print(f"新增 {len(new_days)} 个交易日: {new_days[0].date()} -> {new_days[-1].date()}")

    # 2) 抓取全池日线: tushare本地文件(主源) → 新浪(兜底, 仅补缺票)
    series = {"513100.SS": ref}
    for full in TICKERS:
        if full == "513100.SS":
            continue
        s = _from_local(full)
        if s is None or not all(d in s.index for d in new_days):
            if verbose:
                print(f"  {full} 本地tushare未覆盖全部新增日, 走新浪兜底")
            s2 = fetch_daily(full)
            if s2 is not None:
                s = s2
        if s is None:
            raise RuntimeError(f"数据抓取失败: {full} (双源均失败, 中止保安全)")
        series[full] = s
        if s is not None and verbose:
            time.sleep(0.5)          # 兜底抓取防限流

    # 3) 每票辅助列状态 (从最新行反解; 文件倒序存储, 先按时间正序排)
    dfs = df.sort_values("dt")
    aux = {}
    for full in TICKERS:
        last_px = float(dfs[full].dropna().iloc[-1])
        both = pd.DataFrame({"ln": pd.to_numeric(dfs[full + "_ln"], errors="coerce"),
                             "px": pd.to_numeric(dfs[full], errors="coerce")}).dropna()
        both = both[both.px > 0]
        const = float((both.ln - np.log(both.px)).iloc[-1]) if len(both) else 0.0
        base_col = pd.to_numeric(dfs[full + "_base"], errors="coerce").dropna()
        aux[full] = {
            "const": const,
            "base": float(base_col.iloc[-1]) if len(base_col) else 1.0,
            "a32": float(pd.to_numeric(dfs[full + "_avgs32"], errors="coerce").dropna().iloc[-1]),
            "a128": float(pd.to_numeric(dfs[full + "_avgs128"], errors="coerce").dropna().iloc[-1]),
            "prev_close": last_px,
        }

    # 4) 逐日构造 8 bar (时间序算 EMA, 存储倒序)
    rows = []
    for d in new_days:
        day_rows = {}
        for full in TICKERS:
            s = series[full]
            if d in s.index:
                o_, c_ = float(s.loc[d, "open"]), float(s.loc[d, "close"])
                if not (np.isfinite(o_) and o_ > 0):
                    o_ = c_
                if not (np.isfinite(c_) and c_ > 0):
                    o_ = c_ = aux[full]["prev_close"]
            else:
                o_ = c_ = aux[full]["prev_close"]   # 停牌顺延
            aux[full]["prev_close"] = c_
            day_rows[full] = (o_, c_)
        # EMA 按时间序 (0931->1500)
        for hm in BAR_TIMES:
            for full in TICKERS:
                o_, c_ = day_rows[full]
                px = o_ if hm == "0931" else c_
                ln = np.log(px) + aux[full]["const"]
                aux[full]["a32"] += (ln - aux[full]["a32"]) / 32
                aux[full]["a128"] += (ln - aux[full]["a128"]) / 128
        # 存储倒序 (1500 在前)
        for hm in reversed(BAR_TIMES):
            row = {"trade_date": d.strftime("%Y%m%d") + hm}
            for full in TICKERS:
                o_, c_ = day_rows[full]
                px = o_ if hm == "0931" else c_
                ln = np.log(px) + aux[full]["const"]
                # 注: _ln/_avgs 在各行重复当值(占位延续), 消费者为零
                row[full] = px
                row[full + "_fact"] = 1
                row[full + "_ln"] = ln
                row[full + "_base"] = aux[full]["base"]
                row[full + "_avgs128"] = aux[full]["a128"]
                row[full + "_avgs32"] = aux[full]["a32"]
            rows.append(row)
    new_df = pd.DataFrame(rows)[list(df.columns[:-1])]   # 去掉临时 dt 列, 保列序

    # 5) 接缝校验: 重叠日价格一致性 (防数据源口径漂移污染信号)
    for full in TICKERS:
        s = series[full]
        overlap = [d for d in s.index if d <= last_day][-5:]
        hist = df[df.dt.dt.strftime("%H%M") == "1500"].copy()
        hist["d"] = hist.dt.dt.normalize()
        hist = hist.set_index("d")[full]
        hist = pd.to_numeric(hist, errors="coerce")
        for d in overlap:
            if d in hist.index and hist.loc[d] > 0:
                dev = abs(float(s.loc[d, "close"]) / float(hist.loc[d]) - 1)
                if dev > 0.005:
                    raise RuntimeError(
                        f"接缝校验失败 {full} {d.date()}: 历史={float(hist.loc[d])} "
                        f"新浪={float(s.loc[d, 'close'])} 偏差{dev:.2%} (>0.5%) —— "
                        f"口径漂移, 中止写入防信号污染")

    if check_only:
        print(f"--check 模式: 接缝校验通过, 未写入 ({len(new_days)} 日待更)")
        return new_days
    out = pd.concat([new_df, df.drop(columns=["dt"])], ignore_index=True)
    out.to_csv(YDATA, index=False)
    if verbose:
        print(f"已写入 {YDATA}: +{len(new_df)} 行, 总计 {len(out)} 行")
    return new_days


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查不写入")
    args = ap.parse_args()
    update(check_only=args.check)
