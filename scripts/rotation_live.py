#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动 v3.10 实盘管线 (claude_big) —— 决策孪生: 全量重跑=天然保真

与主线 king_live 的增量状态推进不同, 本策略状态完全由 (价格序列, LLM文档) 决定,
每晚全量重跑 bt_v31 正典配置, 最后一日的装配目标即次日交易指导 ——
实盘决策与回测逐日一致由构造保证 (无孪生漂移风险)。

每晚流程 (--run-daily, 由 quant_levels/holding.py 在 king_live 之后调用):
  1. update_ydata: y-data.csv 全量重建 (tushare复权口径: 银行=common_data前复权,
     ETF=fund_daily×fund_adj; 失败容错沿用昨日数据)
  2. 月末交易日且当月月报缺失 → 先跑月度LLM分析 (monthly_llm_analysis)
  3. 持有B/V/W2仓且本周事件doc缺失 → 生成今日事件级LLM doc (白皮书§九: 衰竭退出/护航)
  4. run_v31_canonical 全量重跑 (target_sink 收集每日装配目标)
  5. 指导文件: output/live/big_directive_{signal_date}.json + big_directive_latest.json
     (同步到 quant_levels/history_output_new/ 供部署平台获取, 与 wz2 同约定)
  6. 追加 output/live/big_live_log.jsonl (日期/净值/目标权重/交易)

月末判定: akshare 交易日历(tool_trade_date_hist_sina, 缓存 output/live/trade_cal.json)
  —— 今日之后首个交易日跨月 = 今日月末; 日历缺失时 fallback: 周五且5个自然日内跨月。

用法:
  python3 scripts/rotation_live.py --run-daily           # 每晚管线
  python3 scripts/rotation_live.py --emit-directives     # 全历史指导文件(回测数据包用)
  python3 scripts/rotation_live.py --check               # 与CLI正典一致性校验
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotation_backtest import TICKERS, load_daily, run_v31_canonical  # noqa: E402
import update_ydata  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(ROOT, "output", "live")
DIRECTIVE_DIR = os.path.join(LIVE_DIR, "directives")
LLM_DIR = os.path.join(ROOT, "output", "llm_monthly")
EVENTS_DIR = os.path.join(ROOT, "output", "llm_events")
# 与 king_live 同款同步位: 部署平台从 quant_levels 仓库取 latest 指导文件
SYNC_DIR = os.path.expanduser("~/local_data/stock_src/quant_levels/history_output_new")
CAL_CACHE = os.path.join(LIVE_DIR, "trade_cal.json")

# 原油501018.SS/160723.SZ 20260814移出候选池(2027退市)
B_TICKERS = ["513180.SS", "588000.SS"]
START = "2013-08-01"   # 正典回测起点 (与 --strategy v31 --start 默认一致, 6标的时代)


# ---------------------------------------------------------------- 日历/日期

def trade_calendar():
    """A股交易日历 (akshare新浪, 缓存7天)。"""
    try:
        if os.path.exists(CAL_CACHE):
            doc = json.load(open(CAL_CACHE))
            if time.time() - doc.get("ts", 0) < 7 * 86400:
                return pd.DatetimeIndex(doc["dates"])
    except Exception:
        pass
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        dates = pd.DatetimeIndex(pd.to_datetime(cal["trade_date"]))
        os.makedirs(LIVE_DIR, exist_ok=True)
        json.dump({"ts": time.time(),
                   "dates": [str(d.date()) for d in dates]},
                  open(CAL_CACHE, "w"))
        return dates
    except Exception as e:
        print(f"交易日历获取失败, fallback 周内近似: {e}")
        return None


def is_month_end(day, cal):
    """今日=本月最后交易日? 日历法优先, 周五跨月法兜底。"""
    if cal is not None:
        fut = cal[cal > day]
        if len(fut):
            return fut[0].month != day.month or fut[0].year != day.year
    return day.weekday() == 4 and (day + pd.Timedelta(days=5)).month != day.month


def next_trading_day(day, cal):
    if cal is not None:
        fut = cal[cal > day]
        if len(fut):
            return fut[0]
    d = day + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


# ---------------------------------------------------------------- LLM 更新

def ensure_monthly_llm(closes, T):
    """月末交易日: 当月月报缺失 → 生成 (链式读上月)。"""
    ym = T.strftime("%Y%m")
    if os.path.exists(os.path.join(LLM_DIR, f"{ym}.json")):
        print(f"月度LLM {ym} 已存在, 跳过")
        return
    import monthly_llm_analysis as mla
    mla._ensure_llm_keys()
    me = [t for t in mla.month_ends(closes.index) if t < T]
    prev = None
    if me:
        p = os.path.join(LLM_DIR, f"{me[-1].strftime('%Y%m')}.json")
        if os.path.exists(p):
            prev = json.load(open(p, encoding="utf-8"))
    print(f"月末交易日 {T.date()}: 生成月度LLM {ym} ...")
    tag, st = mla.analyze_month(closes, T, prev)
    print(f"月度LLM {tag}: {st}")


def ensure_event_llm(closes, T, held_b):
    """持有B轨票时, 本周事件doc缺失 → 生成今日事件级分析 (对齐历史周频惯例)。"""
    if not held_b:
        return
    monday = T - pd.Timedelta(days=T.weekday())
    for fn in os.listdir(EVENTS_DIR) if os.path.isdir(EVENTS_DIR) else []:
        if fn.endswith(".json"):
            try:
                d = pd.Timestamp(fn[:8])
                if d >= monday:
                    print(f"本周事件doc已存在 ({fn[:8]}), 跳过")
                    return
            except Exception:
                pass
    import monthly_llm_analysis as mla
    mla._ensure_llm_keys()
    me = [t for t in mla.month_ends(closes.index) if t < T]
    prev = None
    if me:
        p = os.path.join(LLM_DIR, f"{me[-1].strftime('%Y%m')}.json")
        if os.path.exists(p):
            prev = json.load(open(p, encoding="utf-8"))
    print(f"B轨持仓中 {sorted(held_b)}: 生成事件级LLM {T.strftime('%Y%m%d')} ...")
    tag, st = mla.analyze_month(closes, T, prev, out_dir=EVENTS_DIR,
                                tag=T.strftime("%Y%m%d"))
    print(f"事件级LLM {tag}: {st}")


# ---------------------------------------------------------------- 指导文件

def build_directive(signal_date, exec_date, target, equity, note=""):
    return {
        "strategy": "big_rotation_v3.10",
        "signal_date": signal_date.strftime("%Y%m%d"),
        "exec_date": exec_date.strftime("%Y%m%d"),
        "equity": round(float(equity), 6),
        "note": note,
        "market": {"mode": "big"},
        "targets": [{"code": t, "name": TICKERS[t], "weight": round(w, 4)}
                    for t, w in sorted(target.items(), key=lambda kv: -kv[1])
                    if w > 1e-4],
    }


def write_directive(doc, sync=True):
    os.makedirs(DIRECTIVE_DIR, exist_ok=True)
    p = os.path.join(DIRECTIVE_DIR, f"big_directive_{doc['signal_date']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    latest = os.path.join(LIVE_DIR, "big_directive_latest.json")
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    if sync and os.path.isdir(SYNC_DIR):
        p3 = os.path.join(SYNC_DIR, "big_directive_latest.json")
        with open(p3, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
    return p


# ---------------------------------------------------------------- 主流程

def run_daily():
    os.makedirs(LIVE_DIR, exist_ok=True)
    # 1) 数据更新 (全量重建=复权口径一致; 失败则沿用昨日数据继续, 不中断管线)
    try:
        update_ydata.update()
    except Exception as e:
        print(f"!!! y-data 更新失败, 沿用既有数据继续: {e}")
    # 2) 全量数据 (正典起点切片)
    closes, opens = load_daily()
    closes = closes.loc[START:]
    opens = opens.loc[START:]
    T = closes.index[-1]
    cal = trade_calendar()
    print(f"数据截至 {T.date()}")
    # 3) 月末LLM (先于回测, 保证当日装配用上新月报)
    if is_month_end(T, cal):
        try:
            ensure_monthly_llm(closes, T)
        except Exception as e:
            print(f"月度LLM生成失败(回测沿用既有文档继续): {e}")
    # 4) 首跑: 先看持仓决定是否需要事件doc, 再终跑
    sink = []
    eq, trades, wdf = run_v31_canonical(closes, opens, target_sink=sink)
    held_b = {t for t, w in sink[-1][1].items() if t in B_TICKERS and w > 1e-4}
    if held_b:
        try:
            before = set(os.listdir(EVENTS_DIR)) if os.path.isdir(EVENTS_DIR) else set()
            ensure_event_llm(closes, T, held_b)
            after = set(os.listdir(EVENTS_DIR)) if os.path.isdir(EVENTS_DIR) else set()
            if after - before:   # 新doc可能影响当日判定 → 重跑
                print("新事件doc已生成, 重跑终局装配")
                sink = []
                eq, trades, wdf = run_v31_canonical(closes, opens, target_sink=sink)
        except Exception as e:
            print(f"事件级LLM生成失败(沿用既有文档继续): {e}")
    # 5) 指导文件
    sig_day, target = sink[-1]
    exec_date = next_trading_day(T, cal)
    recent = [tr for tr in trades if pd.Timestamp(tr[0]) >= T - pd.Timedelta(days=10)]
    note = "; ".join(f"{tr[0]} {tr[1]} {tr[2]}" for tr in recent[-5:])
    doc = build_directive(sig_day, exec_date, target, eq.iloc[-1], note)
    path = write_directive(doc, sync=True)
    # 6) live 日志
    rec = {"date": str(T.date()), "equity": round(float(eq.iloc[-1]), 6),
           "target": {t: round(w, 4) for t, w in target.items() if w > 1e-4},
           "exec_date": exec_date.strftime("%Y%m%d")}
    with open(os.path.join(LIVE_DIR, "big_live_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"指导文件: {path}")
    print(f"信号日 {sig_day.date()} → 执行日 {exec_date.date()}, "
          f"净值 {float(eq.iloc[-1]):.4f}")
    for t_ in doc["targets"]:
        print(f"  {t_['code']} {t_['name']}: {t_['weight']:.1%}")
    cash_w = 1 - sum(x["weight"] for x in doc["targets"])
    print(f"  现金: {cash_w:.1%}")


def emit_directives(start="2013-08-01"):
    """全历史指导文件 (PTrade回测数据包用): 每日信号 → 次交易日执行。"""
    os.makedirs(DIRECTIVE_DIR, exist_ok=True)
    closes, opens = load_daily()
    closes = closes.loc[start:]
    opens = opens.loc[start:]
    sink = []
    eq, trades, wdf = run_v31_canonical(closes, opens, target_sink=sink)
    days = list(closes.index)
    n = 0
    for i, (sig_day, target) in enumerate(sink):
        if i + 1 >= len(days):
            break
        doc = build_directive(sig_day, days[i + 1], target, eq.loc[sig_day])
        p = os.path.join(DIRECTIVE_DIR, f"big_directive_{sig_day.strftime('%Y%m%d')}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        n += 1
    # latest = 最后一份 (含末日目标, 供部署)
    sig_day, target = sink[-1]
    doc = build_directive(sig_day, next_trading_day(days[-1], trade_calendar()),
                          target, eq.iloc[-1])
    write_directive(doc, sync=False)
    print(f"已生成 {n} 份历史指导文件 → {DIRECTIVE_DIR}")


def check():
    """与CLI正典一致性校验: run_v31_canonical == --strategy v31 全默认参数。"""
    closes, opens = load_daily()
    closes = closes.loc[START:]
    opens = opens.loc[START:]
    eq1, tr1, _ = run_v31_canonical(closes, opens)
    print(f"canonical: 终值 {float(eq1.iloc[-1]):.6f}, 交易 {len(tr1)} 笔")
    print("请与 `python3 scripts/rotation_backtest.py --strategy v31` 输出对照 "
          "(期望终值一致; v3.10定稿 +994.3% → 终值约10.94, 随数据日更自然增长)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-daily", action="store_true")
    ap.add_argument("--emit-directives", action="store_true")
    ap.add_argument("--start", default="2013-08-01")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.run_daily:
        run_daily()
    elif args.emit_directives:
        emit_directives(args.start)
    elif args.check:
        check()
    else:
        ap.print_help()
