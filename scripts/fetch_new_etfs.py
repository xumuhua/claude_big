#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三只新ETF历史数据抓取 (20260822 哥哥任务: 588060/159920/159952 合入 claude_big)

口径与 fetch_stock_price.py::fetch_big_pool() 完全一致:
  tushare pro.fund_daily 原始价 × pro.fund_adj 复权因子 / 最新因子 = 前复权
输出: ~/local_data/common_data/big_pool_extra_price.csv (schema同big_pool_price.csv)
不改生产 fetch_stock_price.py / BIG_POOL —— 本脚本为实验分支专用数据源。
"""
import os
import sys
import time

import pandas as pd
import tushare as ts

sys.path.insert(0, "/home/claude/local_data/stock_src/quant_levels")
from config import TUSHARE_TOKEN  # noqa: E402

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

NEW_POOL = {
    "588060.SH": "588060.SS",   # 科创板ETF(工银科创50) → star类(与588000同指数)
    "159920.SZ": "159920.SZ",   # 恒生ETF(华夏, 恒生指数) → hk类
    "159952.SZ": "159952.SZ",   # 创业板ETF(广发, 创业板指) → chinext类
}
OUT = os.path.expanduser("~/local_data/common_data/big_pool_extra_price.csv")


def fetch_one(ts_code, ycode, end):
    df = adj = None
    for k in range(3):
        try:
            df = pro.fund_daily(ts_code=ts_code, start_date='20100101', end_date=end,
                                fields='trade_date,open,high,low,close,vol,amount')
            adj = pro.fund_adj(ts_code=ts_code, start_date='20100101', end_date=end)
            break
        except Exception as e:
            print(f"  {ts_code} 第{k + 1}次失败: {str(e)[:80]}")
            time.sleep(2 * (k + 1))
    if df is None or len(df) == 0:
        print(f"  错误: {ts_code} fund_daily 无数据")
        return None
    df['trade_date'] = df['trade_date'].astype(str)
    if adj is not None and len(adj):
        adj['trade_date'] = adj['trade_date'].astype(str)
        adj = adj.sort_values('trade_date')
        f_last = float(adj.adj_factor.iloc[-1])
        df = df.merge(adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
        df['adj_factor'] = df['adj_factor'].astype(float).bfill().ffill()
        df['open'] = df['open'] * df['adj_factor'] / f_last
        df['close'] = df['close'] * df['adj_factor'] / f_last
        n_ev = int((df.adj_factor.diff().abs() > 1e-9).sum())
        print(f"  {ycode}: {len(df)} 行 {df.trade_date.min()}->{df.trade_date.max()}, "
              f"复权事件 {n_ev} 次, 末因子 {f_last:.4f}")
    else:
        print(f"  警告: {ycode} fund_adj 无数据, 用原始价 ({len(df)} 行)")
    return df[['trade_date', 'open', 'close']].assign(code=ycode)


def main():
    import datetime as _dt
    end = _dt.date.today().strftime('%Y%m%d')
    frames = []
    for ts_code, ycode in NEW_POOL.items():
        f = fetch_one(ts_code, ycode, end)
        if f is not None:
            frames.append(f)
        time.sleep(0.4)
    if not frames:
        raise RuntimeError("全部获取失败")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"已写入 {OUT}: {len(out)} 行")


if __name__ == "__main__":
    main()
