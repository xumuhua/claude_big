#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF/指数轮动回测引擎 (claude_big 策略线)

定位: 主线空仓期的资金利用补充轨 —— 全天候迷你池(纳指/恒生科技/科创50/黄金/原油/四大行)动量轮动。

数据: y-data.csv (8bar/日: 0931,1000,1031,1100,1331,1400,1431,1500; 上市前回填-1.0)
  - 主列=分红复权价, 但 513100 2022-01-14 1拆5 未复权 → 引擎内自动检测(|日收益|>30%)并修复
  - 信号用 15:00 收盘价, 执行用次日 09:31 首bar价 (T+1 开盘成交, 与主线 ptrade_wz2 9:30 买入一致)

策略: 双动量(相对+绝对)轮动, 默认配方(20260805 第一轮优化结论):
  - score = (0.5*ret20 + 0.5*ret60) / 20日年化波动率
  - 每10个交易日调仓: 全部 score>0 且过类上限(每类1只)的标的一律持有, 1/波动率加权, 单票≤35%
  - 每日风控: 持仓 score < -0.2(滞后带宽) → 次日开盘离场并冷却10日
  - 净值熔断: 策略净值 < MA150 → 空仓观望, 收回上方 → 恢复 (慢熊磨损失血的总闸)
  - LLM退出侧(20260806 v2采纳): 豆包月度阶段判定=下降 → 持仓次日离场(只卖不买)
用途:
  python3 scripts/rotation_backtest.py                       # 默认配方(推荐)
  python3 scripts/rotation_backtest.py --eq-ma 0             # 关熔断对照
  python3 scripts/rotation_backtest.py --grid                # 小网格稳健性对照
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "y-data.csv")

TICKERS = {
    "513100.SS": "纳指ETF", "513180.SS": "恒生科技", "588000.SS": "科创50",
    "518800.SS": "黄金ETF", "501018.SS": "南方原油", "160723.SZ": "嘉实原油",
    "601398.SS": "工行", "601988.SS": "中行", "601939.SS": "建行", "601288.SS": "农行",
}
# ETF(含LOF)无印花税; 银行股卖出有印花税 → 分类成本
STOCKS = {"601398.SS", "601988.SS", "601939.SS", "601288.SS"}
# 资产类别(类内高度同质, 类上限1只): 油/行/股指/金
CLASSES = {
    "513100.SS": "us", "513180.SS": "hk", "588000.SS": "star",
    "518800.SS": "gold", "501018.SS": "oil", "160723.SZ": "oil",
    "601398.SS": "bank", "601988.SS": "bank", "601939.SS": "bank", "601288.SS": "bank",
}
# v3 分轨(用户20260806): A=低波复利轨(趋势持有到反转), B=高波周期轨(底部反转等待)
A_TRACK = {"513100.SS": 0.50, "518800.SS": 0.50,
           "601398.SS": 0.125, "601988.SS": 0.125, "601939.SS": 0.125, "601288.SS": 0.125}
B_TRACK = {"513180.SS": 0.15, "588000.SS": 0.15, "501018.SS": 0.125, "160723.SZ": 0.125}

NAME = os.path.basename(DATA)


def load_daily():
    """读分钟采样数据 → (日收盘价矩阵, 日开盘价矩阵=09:31bar), 拆分修复后"""
    df = pd.read_csv(DATA, dtype={"trade_date": str})
    df["dt"] = pd.to_datetime(df.trade_date, format="mixed")
    df = df.sort_values("dt").reset_index(drop=True)
    df["d"] = df.dt.dt.date
    df["hm"] = df.dt.dt.strftime("%H%M")
    close = df[df.hm == "1500"].copy()
    close.index = close.dt.dt.normalize()
    openp = df[df.hm == "0931"].copy()
    openp.index = openp.dt.dt.normalize()
    closes, opens = {}, {}
    for t in TICKERS:
        c = close[t].copy()
        c[c <= 0] = np.nan          # 上市前回填-1.0 → NaN
        o = openp[t].copy()
        o[o <= 0] = np.nan
        # 拆分修复: |日收益|>30% 不可能是真实交易(A股10%/20%限制) → 前复权
        r = c.pct_change()
        for day in r[r.abs() > 0.30].index:
            ratio = c.loc[day] / c.shift(1).loc[day]
            c.loc[c.index < day] *= ratio
            o.loc[o.index < day] *= ratio
        closes[t] = c
        opens[t] = o
    return pd.DataFrame(closes), pd.DataFrame(opens)


def bt(closes, opens, lookbacks, weights, topk, gate, reb_days, cost_etf, cost_stock,
       vol_adj=False, exit_daily=False, class_cap=0, trend_ma=0, trail_stop=None,
       hyst=0.0, cooldown=0, inc_margin=0.0, weight_mode="equal", max_w=0.0, eq_ma=0,
       llm_mode="off", llm_dir=None, llm_cap=0.5, llm_conf=0.6, llm_bottom_w=0.5, llm_exit_conf=0.0):
    """现金+份额双账本; 信号日收盘打分, 次日09:31成交; 留仓不收费, 买卖各收分类费率

    vol_adj:    score /= 20日波动率 (避免高波原油霸占排名)
    exit_daily: 每日检查持仓 score<=gate 或跌破趋势MA → 次日开盘离场 (不等调仓日)
    class_cap:  每资产类最多持有只数 (0=不限; 油×2/行×4 同质, 建议1)
    trend_ma:   >0 时要求 收盘价 > MA(trend_ma) 才允许持有 (绝对趋势过滤)
    trail_stop: <0 时启用持仓移动止损: 收盘价较买入后峰值回撤 <= trail_stop → 次日离场
    hyst:       滞后带宽(防闸门抖动): 每日离场需 score < gate - hyst (入场仍按 gate)
    cooldown:   风控离场后 N 个交易日内该标的不可再入场 (防同票反复横跳)
    inc_margin: 在位优势(防调仓日同质换仓): 持仓排名分 += inc_margin, 挑战者须明显更强才换
    topk=0:     不限制只数 —— 全部过闸标的都持有(类上限仍生效), 权重按 weight_mode
    weight_mode: equal=等权, invvol=1/波动率加权
    max_w:      单标的权重上限(超出部分留现金), 0=不限
    eq_ma:      >0 时启用净值曲线熔断: 策略净值收盘 < 其MA(eq_ma) → 次日起空仓,
                收回MA上方 → 次日恢复。权重皆为比例制, 空仓期信号照常演进, 故后处理精确等价。
    llm_mode:   off=不用;
                v1(已证伪档): veto=看空禁入场; scale=risk_off月仓位×llm_cap; both=两者
                v2(阶段判定): phase=全开(早鸟+veto+下降退出); phase-bottom=仅早鸟;
                              phase-exit=仅冲高衰竭veto+下降退出
                v2 阶段→动作映射(用户20260806方法论):
                  筑底+逻辑强/中+conf≥llm_conf → 早鸟入场(绕过score闸门, 权重×llm_bottom_w)
                  冲高+逻辑衰竭 → 禁入场 (强+持续的冲高不veto——强逻辑行情会延续)
                  下降 → 禁入场 + 持仓次日退出(比score确认更早)
                LLM 月度产物在 llm_dir 下({YYYYMM}.json, 月末晚间生成), asof<=当日即生效,
                与信号 T+1 纪律一致。
    """
    days = closes.index.intersection(opens.index)
    score = sum(w * (closes / closes.shift(lb) - 1) for w, lb in zip(weights, lookbacks))
    if vol_adj:
        vol = closes.pct_change().rolling(20).std() * np.sqrt(244)
        score = score / vol.replace(0, np.nan)
    listed = closes.notna()
    cshift = closes.shift(1)
    vol20 = closes.pct_change().rolling(20).std() * np.sqrt(244)
    ma = closes.rolling(trend_ma).mean() if trend_ma else None
    reb_set = set(days[i] for i in range(0, len(days), reb_days))

    # LLM 月度产物时间线: [(asof, doc)] 按时间升序; 月末晚间生成, asof<=当日生效
    llm_tl = []
    if llm_mode != "off" and llm_dir and os.path.isdir(llm_dir):
        for fn in sorted(os.listdir(llm_dir)):
            if fn.endswith(".json"):
                try:
                    doc = json.load(open(os.path.join(llm_dir, fn), encoding="utf-8"))
                    llm_tl.append((pd.Timestamp(doc["asof"]), doc))
                except Exception:
                    pass
        llm_tl.sort(key=lambda x: x[0])

    cash, pos = 1.0, {}             # pos: ticker -> 份额
    peak = {}                       # ticker -> 买入后最高收盘价 (移动止损用)
    early_entry = {}                # ticker -> 早鸟入场价 (v2筑底逆势仓: 持有保护+硬止损)
    cool = {}                       # ticker -> 冷却截止日 (风控离场后)
    pending = None                  # 待执行目标权重
    curve, trades = [], []

    llm_state = {"veto": set(), "regime": None, "idx": -1,
                 "early": set(), "exit": set()}     # v2: 早鸟/下降退出集合

    def eligible(t, day):
        """t 在 day 是否允许(再)入场"""
        if cooldown and day <= cool.get(t, day - pd.Timedelta(days=1)):
            return False
        # v2 早鸟: 筑底+强/中逻辑 → 绕过 score/MA 闸门 (逆势入场, 权重另限)
        if llm_mode in ("phase", "phase-bottom") and t in llm_state["early"]:
            return True
        if llm_mode in ("veto", "both") and t in llm_state["veto"]:  # v1 看空 veto
            return False
        if llm_mode in ("phase", "phase-exit") and t in llm_state["veto"]:  # v2 下降+冲高衰竭 veto
            return False
        if llm_mode == "phase-top" and t in llm_state["veto_top"]:   # v2 仅冲高衰竭 veto
            return False
        s = score.loc[day, t]
        if not np.isfinite(s) or s <= gate:
            return False
        if ma is not None:
            m = ma.loc[day, t]
            if not np.isfinite(m) or closes.loc[day, t] <= m:
                return False
        return True

    def holdable(t, day):
        """t 在 day 是否允许继续持有 (滞后带宽: 需明显破闸才离场)"""
        if llm_mode in ("phase", "phase-exit", "phase-top", "phase-sell") and t in llm_state["exit"]:
            return False                              # v2: 阶段=下降 → 不可持有(调仓日同样生效)
        if llm_mode in ("phase", "phase-bottom") and t in early_entry and t in llm_state["early"]:
            return True                               # v2: 早鸟持有保护(筑底判定仍有效时不被震出)
        s = score.loc[day, t]
        if not np.isfinite(s) or s <= gate - hyst:
            return False
        if ma is not None:
            m = ma.loc[day, t]
            if not np.isfinite(m) or closes.loc[day, t] <= m:
                return False
        return True

    for i, day in enumerate(days):
        c, o, cp = closes.loc[day], opens.loc[day], cshift.loc[day]
        # 0) LLM 月度状态推进 (asof<=当日 的最新一份生效)
        while llm_state["idx"] + 1 < len(llm_tl) and llm_tl[llm_state["idx"] + 1][0] <= day:
            llm_state["idx"] += 1
            doc = llm_tl[llm_state["idx"]][1]
            if doc.get("version") == 2:
                # v2 阶段schema: veto拆两档(下降/冲高衰竭)便于消融
                llm_state["regime"] = None
                llm_state["early"] = {
                    t for t, a in doc.get("assets", {}).items()
                    if a.get("phase") == "筑底" and a.get("logic_strength") in ("强", "中")
                    and float(a.get("confidence", 0) or 0) >= llm_conf
                }
                llm_state["veto_dn"] = {
                    t for t, a in doc.get("assets", {}).items() if a.get("phase") == "下降"
                }
                llm_state["exit"] = {
                    t for t, a in doc.get("assets", {}).items()
                    if a.get("phase") == "下降"
                    and float(a.get("confidence", 0) or 0) >= llm_exit_conf
                }
                llm_state["veto_top"] = {
                    t for t, a in doc.get("assets", {}).items()
                    if a.get("phase") == "冲高" and a.get("logic_durability") == "衰竭"
                }
                llm_state["veto"] = llm_state["veto_dn"] | llm_state["veto_top"]
            else:
                # v1 评级schema
                llm_state["regime"] = doc.get("regime")
                llm_state["early"], llm_state["exit"] = set(), set()
                llm_state["veto_top"], llm_state["veto_dn"] = set(), set()
                llm_state["veto"] = {
                    t for t, a in doc.get("assets", {}).items()
                    if a.get("rating") == "看空" and float(a.get("confidence", 0) or 0) >= llm_conf
                }
        # 1) 开盘执行昨日信号
        if pending is not None:
            fee_of = lambda t: cost_stock if t in STOCKS else cost_etf
            px_of = lambda t: o[t] if np.isfinite(o.get(t, np.nan)) else cp.get(t, np.nan)
            total = cash + sum(n * px_of(t) for t, n in pos.items() if np.isfinite(px_of(t)))
            fees, new_pos = 0.0, {}
            # 旧持仓: 目标内留仓, 目标外卖出
            for t, n in pos.items():
                px = px_of(t)
                if not np.isfinite(px):
                    new_pos[t] = n
                    continue
                if t in pending:
                    new_pos[t] = n
                else:
                    fees += n * px * fee_of(t)
                    peak.pop(t, None)
                    early_entry.pop(t, None)
                    trades.append((str(day.date()), "SELL", TICKERS[t], f"{n * px / total:.1%}"))
            # 新买入: 按目标权重分配
            for t, w in pending.items():
                if t in new_pos:
                    continue
                px = px_of(t)
                if not np.isfinite(px) or px <= 0:
                    continue
                alloc = total * w
                new_pos[t] = alloc * (1 - fee_of(t)) / px
                peak[t] = px
                if llm_mode in ("phase", "phase-bottom") and t in llm_state["early"]:
                    early_entry[t] = px
                fees += alloc * fee_of(t)
                trades.append((str(day.date()), "BUY", TICKERS[t], f"{w:.0%}"))
            mv_open = sum(n * px_of(t) for t, n in new_pos.items() if np.isfinite(px_of(t)))
            cash = total - mv_open - fees
            pos = new_pos
            pending = None
        # 2) 日终估值 + 移动止损峰值更新
        mv = sum(n * c.get(t, np.nan) for t, n in pos.items() if np.isfinite(c.get(t, np.nan)))
        curve.append((day, cash + mv))
        for t in pos:
            if np.isfinite(c.get(t, np.nan)):
                peak[t] = max(peak.get(t, 0.0), c[t])
        # 3) 收盘信号
        if day in reb_set:
            s = score.loc[day]
            cand = []
            for t in TICKERS:
                if not (listed.loc[day, t] and np.isfinite(s.get(t, np.nan))):
                    continue
                if t in pos:
                    if not holdable(t, day):
                        continue
                    eff = s[t] + inc_margin       # 在位优势
                else:
                    if not eligible(t, day):
                        continue
                    eff = s[t]
                # v2 早鸟: score 为负也保留候选资格, 按微正分参与排名(靠后可入选)
                if llm_mode in ("phase", "phase-bottom") and t in llm_state["early"]:
                    eff = max(s[t] if np.isfinite(s[t]) else 0.0, 0.01) + (inc_margin if t in pos else 0.0)
                cand.append((t, eff))
            cand.sort(key=lambda x: -x[1])
            picks, cls_cnt = [], {}
            for t, v in cand:
                cl = CLASSES[t]
                if class_cap and cls_cnt.get(cl, 0) >= class_cap:
                    continue
                picks.append(t)
                cls_cnt[cl] = cls_cnt.get(cl, 0) + 1
                if topk and len(picks) >= topk:
                    break
            # 权重: 等权或1/波动率; max_w封顶, 余量留现金; 早鸟票权重上限×llm_bottom_w
            if picks:
                if weight_mode == "invvol":
                    iv = {t: 1.0 / max(vol20.loc[day, t], 0.05) for t in picks}
                    tot = sum(iv.values())
                    new_target = {t: iv[t] / tot for t in picks}
                else:
                    new_target = {t: 1.0 / len(picks) for t in picks}
                if max_w > 0:
                    new_target = {t: min(w, max_w) for t, w in new_target.items()}
                if llm_mode in ("phase", "phase-bottom"):
                    bw = max_w * llm_bottom_w if max_w > 0 else llm_bottom_w
                    new_target = {t: min(w, bw) if t in llm_state["early"] and t not in pos else w
                                  for t, w in new_target.items()}
                # LLM regime 仓位缩放: risk_off 月总仓位×cap, 余量现金
                if llm_mode in ("scale", "both") and llm_state["regime"] == "risk_off":
                    new_target = {t: w * llm_cap for t, w in new_target.items()}
            else:
                new_target = {}
            # 持仓不变则不触发交易
            if set(new_target) != set(pos) or pending is not None:
                pending = new_target
        elif (exit_daily or trail_stop or llm_mode in ("phase", "phase-exit", "phase-top", "phase-sell")) and pos:
            # 非调仓日的每日风控: 持仓明显破闸(滞后带宽)/破趋势/移动止损/LLM下降判定 → 次日离场
            bad = []
            for t in pos:
                if not (listed.loc[day, t] and np.isfinite(c.get(t, np.nan))):
                    continue
                if t in early_entry and c[t] / early_entry[t] - 1 <= -0.08:
                    bad.append(t)                     # v2: 早鸟成本锚定硬止损-8%(逆势仓纪律)
                    trades.append((str(day.date()), "EARLY-STOP", TICKERS[t],
                                   f"{c[t] / early_entry[t] - 1:.1%}"))
                elif llm_mode in ("phase", "phase-exit", "phase-top", "phase-sell") and t in llm_state["exit"]:
                    bad.append(t)
                    trades.append((str(day.date()), "LLM-EXIT", TICKERS[t], "阶段=下降"))
                elif exit_daily and not holdable(t, day):
                    bad.append(t)
                    trades.append((str(day.date()), "RISK-OFF", TICKERS[t], "score破闸/破趋势"))
                elif trail_stop and peak.get(t) and c[t] / peak[t] - 1 <= trail_stop:
                    bad.append(t)
                    trades.append((str(day.date()), "TRAIL-STOP", TICKERS[t],
                                   f"{c[t] / peak[t] - 1:.1%}"))
            if bad:
                if cooldown:
                    for t in bad:
                        cool[t] = days[min(i + cooldown, len(days) - 1)]
                kept = [t for t in pos if t not in bad]
                pending = {t: 1.0 / len(kept) for t in kept} if kept else {}

    eq = pd.Series(dict(curve)).sort_index()
    if eq_ma and len(eq) > eq_ma:
        # 净值熔断(精确后处理): 比例权重下空仓期收益=0, 持仓期收益=满仓曲线收益
        ma_line = eq.rolling(eq_ma).mean()
        inv = (eq >= ma_line).shift(1).fillna(False)
        r = eq.pct_change().fillna(0.0)
        real = (1 + r * inv).cumprod()
        real.iloc[0] = 1.0
        # 熔断状态变化点记入交易日志
        state = inv.astype(bool)
        for d in state.index[state != state.shift(1).fillna(state.iloc[0])]:
            trades.append((str(d.date()), "CIRCUIT-ON" if state[d] else "CIRCUIT-OFF", "", ""))
        eq = real
    return eq, trades


# ---------------------------------------------------------------------------
# v3.1: 不可动核心+边缘增强 (20260806数据修正版)
#   数据结论: 纳指/黄金的任何退出规则都输给永不卖出(空窗复亏纳指1.69x/黄金1.11x)
#   核心: 纳指50%永持 + 黄金50%永持(可让渡≤20%给B轨)
#   边缘: B轨底部反转(用户规则)从黄金袖珍出资; 黄金自身失势时换银行防御
# ---------------------------------------------------------------------------

def bt_v31(closes, opens, llm_dir=None, cost_etf=0.0005, cost_stock=0.001,
           b_arm_dd=-0.25, b_bottom_age=20, b_stop=-0.12, b_exit_ma=60,
           b_total_cap=0.20, gold_buf=0.02, use_llm_exit=True, dust=0.02,
           ndx_w=0.50, gold_w=0.50,
           oh_ext=0.275, oh_r20=0.12, stag_ext=0.12, stag_r250=0.20, stag_r60=0.0,
           oh_re_dd=-0.08, use_oh=True, oh_cool=20, use_stag=False,
           oh_re_age=20, oh_re_slope=False,
           sb_days=0, sb_re_days=5, sb_to="cash"):
    """sb_days>0: 慢性破位换防(用户20260806迭代轮1)——核心票连续sb_days日收于
    MA250×0.98下方 → 袖珍转sb_to(cash/banks); 重新站上MA250满sb_re_days日 → 回归。
    与OH止盈正交: OH是过热前瞻止盈, SB是慢熊确认换防(2022型)。"""
    """过热/滞涨收紧止盈(用户20260806): 核心常态永不卖出, 过热武装态下 diff5<0 坚决止盈。
    oh: 冲顶武装= 偏离MA250≥oh_ext ∧ ret20≥oh_r20 (泡沫加速)
    stag: 滞涨武装= 偏离≥stag_ext ∧ ret250≥stag_r250 ∧ ret60≤stag_r60 (长牛后动力衰竭)
    止盈后回补: 收盘>卖出日收盘(判错认错) 或 回撤≥|oh_re_dd|且低点≥10日且收盘>MA20(调整充分)"""
    days = closes.index.intersection(opens.index)
    listed = closes.notna()
    cshift = closes.shift(1)
    ma250 = closes.rolling(250).mean()
    ma20 = closes.rolling(20).mean()
    ma20_slope = ma20 / ma20.shift(20) - 1
    maX = closes.rolling(b_exit_ma).mean()
    hi250 = closes.rolling(250).max()
    GOLD, NDX = "518800.SS", "513100.SS"
    BANKS = ["601398.SS", "601988.SS", "601939.SS", "601288.SS"]

    low_age = {}
    for t in B_TRACK:
        px = closes[t].values
        age = np.full(len(days), 10**9)
        for i in range(len(days)):
            if not np.isfinite(px[i]):
                continue
            j0 = max(0, i - 249)
            w = px[j0:i + 1]
            if np.isfinite(w).any():
                age[i] = i - (j0 + int(np.nanargmin(w)))
        low_age[t] = age

    llm_tl = []
    if llm_dir and os.path.isdir(llm_dir):
        for fn in sorted(os.listdir(llm_dir)):
            if fn.endswith(".json"):
                try:
                    doc = json.load(open(os.path.join(llm_dir, fn), encoding="utf-8"))
                    if doc.get("version") == 2:
                        llm_tl.append((pd.Timestamp(doc["asof"]), doc))
                except Exception:
                    pass
        llm_tl.sort(key=lambda x: x[0])
    llm_state = {"idx": -1, "exit": set(), "block": set()}
    b_pending_low = {}             # 信号日暂存底部参考低
    sb_streak = {}                 # t -> 连续低于MA250×0.98天数
    sb_out = {}                    # t -> True (换防中)
    sb_re = {}                     # t -> 重新站上MA250天数
    oh_ref = {}                    # t -> 止盈卖出日收盘价 (回补参照)
    oh_low_age = {}                # t -> 止盈后低点计数器用最近低
    oh_armed = {}                  # t -> True (武装闩锁: 直到真调整<MA60或触发才解除)
    oh_coold = {}                  # t -> 冷却截止日 (回补后oh_cool日内不再触发)
    ma20c = closes.rolling(20).mean()
    ma60c = closes.rolling(60).mean()

    cash, pos = 1.0, {}
    b_entry = {}                   # t -> {"px","low","est"} 入场价/底部参考低/趋势已成立
    gold_off = False               # 黄金失势状态(滞回)
    pending = None
    curve, trades = [], []

    for i, day in enumerate(days):
        c, o, cp = closes.loc[day], opens.loc[day], cshift.loc[day]
        while llm_state["idx"] + 1 < len(llm_tl) and llm_tl[llm_state["idx"] + 1][0] <= day:
            llm_state["idx"] += 1
            doc = llm_tl[llm_state["idx"]][1]
            llm_state["exit"] = {t for t, a in doc.get("assets", {}).items()
                                 if a.get("phase") == "下降"}
            llm_state["block"] = llm_state["block"] & llm_state["exit"]
        # 1) 开盘执行
        if pending is not None:
            fee_of = lambda t: cost_stock if t in STOCKS else cost_etf
            px_of = lambda t: o[t] if np.isfinite(o.get(t, np.nan)) else cp.get(t, np.nan)
            total = cash + sum(n * px_of(t) for t, n in pos.items() if np.isfinite(px_of(t)))
            fees, new_pos = 0.0, {}
            for t, n in pos.items():
                px = px_of(t)
                if not np.isfinite(px):
                    new_pos[t] = n
                    continue
                if t in pending:
                    cur_mv, tgt_mv = n * px, total * pending[t]
                    fees += abs(tgt_mv - cur_mv) * fee_of(t) * 0.5
                    new_pos[t] = tgt_mv / px
                else:
                    fees += n * px * fee_of(t)
                    b_entry.pop(t, None)
            for t, w in pending.items():
                if t in pos:
                    continue
                px = px_of(t)
                if not np.isfinite(px) or px <= 0:
                    continue
                new_pos[t] = total * w * (1 - fee_of(t)) / px
                fees += total * w * fee_of(t)
                if t in B_TRACK:
                    b_entry[t] = {"px": px, "low": b_pending_low.pop(t, None), "est": False}
            mv_open = sum(n * px_of(t) for t, n in new_pos.items() if np.isfinite(px_of(t)))
            cash = total - mv_open - fees
            pos = new_pos
            pending = None
        # 2) 日终估值
        mv = sum(n * c.get(t, np.nan) for t, n in pos.items() if np.isfinite(c.get(t, np.nan)))
        curve.append((day, cash + mv))
        # 3) 收盘目标装配
        target = {}
        # 过热/滞涨收紧止盈状态机 (核心两票)
        if use_oh:
            for t in (NDX, GOLD):
                if not (listed.loc[day, t] and np.isfinite(ma250.loc[day, t])):
                    continue
                v = c[t]
                ext = v / ma250.loc[day, t] - 1
                r5 = v / cshift[t] ** 0 if False else None
                px5 = closes[t].iloc[max(0, i - 5):i + 1]
                diff5 = v / px5.iloc[0] - 1 if len(px5) >= 6 else 0.0
                r20 = v / closes[t].iloc[max(0, i - 20)] - 1 if i >= 20 else 0.0
                r60 = v / closes[t].iloc[max(0, i - 60)] - 1 if i >= 60 else 0.0
                r250 = v / closes[t].iloc[max(0, i - 250)] - 1 if i >= 250 else 0.0
                in_pos = t in pos
                # 武装闩锁: 极端态进入, 真调整(破MA60)才解除
                if ext >= oh_ext and r20 >= oh_r20:
                    oh_armed[t] = "冲顶"
                elif use_stag and ext >= stag_ext and r250 >= stag_r250 and r60 <= stag_r60:
                    oh_armed.setdefault(t, "滞涨")
                if np.isfinite(ma60c.loc[day, t]) and v < ma60c.loc[day, t]:
                    oh_armed.pop(t, None)
                if in_pos and t not in oh_ref and oh_armed.get(t) and day > oh_coold.get(t, pd.Timestamp.min):
                    if diff5 < 0:
                        oh_ref[t] = v
                        tag = oh_armed.pop(t)
                        trades.append((str(day.date()), "OH-OUT", TICKERS[t],
                                       f"{tag} ext{ext:+.0%} r20{r20:+.0%} r60{r60:+.0%} diff5{diff5:+.1%}"))
                elif t in oh_ref:
                    # 止盈后跟踪低点
                    lo = oh_low_age.get(t, [v, 0])
                    if v < lo[0]:
                        oh_low_age[t] = [v, 0]
                    else:
                        oh_low_age[t] = [lo[0], lo[1] + 1]
                    back_newhigh = v > oh_ref[t]
                    retrace = v / oh_ref[t] - 1 <= oh_re_dd
                    bottomed = oh_low_age[t][1] >= oh_re_age and np.isfinite(ma20c.loc[day, t]) and v > ma20c.loc[day, t]
                    if bottomed and oh_re_slope:
                        m20s = ma20c[t].iloc[max(0, i - 20):i + 1]
                        bottomed = bottomed and len(m20s) >= 21 and ma20c[t].iloc[-1] > m20s.iloc[0]
                    if back_newhigh or (retrace and bottomed):
                        why = "创新高认错回补" if back_newhigh else "调整充分回补"
                        trades.append((str(day.date()), "OH-IN", TICKERS[t],
                                       f"{why} 止盈价{oh_ref[t]:.2f}现{v:.2f}"))
                        del oh_ref[t]
                        oh_low_age.pop(t, None)
                        oh_coold[t] = days[min(i + oh_cool, len(days) - 1)]   # 回补后冷却
        # 纳指: 永持 ndx_w (过热止盈/慢性破位例外)
        if listed.loc[day, NDX]:
            # 慢性破位状态机
            if sb_days and np.isfinite(ma250.loc[day, NDX]):
                if c[NDX] < ma250.loc[day, NDX] * 0.98:
                    sb_streak[NDX] = sb_streak.get(NDX, 0) + 1
                    sb_re[NDX] = 0
                else:
                    sb_streak[NDX] = 0
                    sb_re[NDX] = sb_re.get(NDX, 0) + 1
                if not sb_out.get(NDX) and sb_streak[NDX] >= sb_days:
                    sb_out[NDX] = True
                    trades.append((str(day.date()), "SB-OUT", TICKERS[NDX],
                                   f"慢破{sb_streak[NDX]}日→{sb_to}"))
                elif sb_out.get(NDX) and sb_re[NDX] >= sb_re_days:
                    sb_out[NDX] = False
                    trades.append((str(day.date()), "SB-IN", TICKERS[NDX], "收复MA250回归"))
            blocked = NDX in oh_ref or sb_out.get(NDX)
            if NDX not in pos and not blocked:
                trades.append((str(day.date()), "CORE-IN", TICKERS[NDX], f"{ndx_w:.0%}永持"))
            if not blocked:
                target[NDX] = ndx_w
            elif sb_out.get(NDX) and sb_to == "banks":
                for b in ["601398.SS", "601988.SS", "601939.SS", "601288.SS"]:
                    if listed.loc[day, b] and np.isfinite(ma250.loc[day, b]) and c[b] > ma250.loc[day, b]:
                        target[b] = target.get(b, 0) + ndx_w / 4
        # 黄金失势滞回
        if listed.loc[day, GOLD] and np.isfinite(ma250.loc[day, GOLD]):
            if not gold_off and c[GOLD] < ma250.loc[day, GOLD] * (1 - gold_buf):
                gold_off = True
                trades.append((str(day.date()), "GOLD-OFF", TICKERS[GOLD], "失势换银行"))
            elif gold_off and c[GOLD] > ma250.loc[day, GOLD] * (1 + gold_buf):
                gold_off = False
                trades.append((str(day.date()), "GOLD-ON", TICKERS[GOLD], "复势回黄金"))
        # B轨: 武装→触发→持有/退出
        b_active = {}
        for t, cap in B_TRACK.items():
            if not (listed.loc[day, t] and np.isfinite(ma250.loc[day, t])):
                continue
            in_pos = t in pos
            if not in_pos:
                if t in llm_state["block"]:
                    continue
                armed = np.isfinite(hi250.loc[day, t]) and c[t] / hi250.loc[day, t] - 1 <= b_arm_dd
                trig = (armed and low_age[t][i] >= b_bottom_age
                        and np.isfinite(ma20.loc[day, t]) and c[t] > ma20.loc[day, t]
                        and np.isfinite(ma20_slope.loc[day, t]) and ma20_slope.loc[day, t] > 0)
                if trig:
                    b_active[t] = cap
                    b_pending_low[t] = float(closes[t].iloc[max(0, i - 249):i + 1].min())
                    trades.append((str(day.date()), "B-IN", TICKERS[t],
                                   f"dd{c[t] / hi250.loc[day, t] - 1:.0%}/低点{low_age[t][i]}日"))
            else:
                be = b_entry.get(t, {})
                # 趋势成立标记: 收盘曾站上MA(b_exit_ma)
                if np.isfinite(maX.loc[day, t]) and c[t] > maX.loc[day, t] and be:
                    be["est"] = True
                new_low = be and be.get("low") is not None and c[t] < be["low"]
                trend_end = be.get("est") and np.isfinite(maX.loc[day, t]) and c[t] < maX.loc[day, t]
                llm_down = use_llm_exit and t in llm_state["exit"]
                stop = be and c[t] / be["px"] - 1 <= b_stop
                if not new_low and not trend_end and not llm_down and not stop:
                    b_active[t] = cap
                else:
                    if llm_down:
                        llm_state["block"].add(t)
                    reason = ("筑底失败创新低" if new_low else "趋势结束" if trend_end
                              else "LLM下降" if llm_down else "硬止损")
                    trades.append((str(day.date()), "B-OUT", TICKERS[t], reason))
        b_sum = sum(b_active.values())
        if b_sum > b_total_cap:
            b_active = {t: w * b_total_cap / b_sum for t, w in b_active.items()}
            b_sum = b_total_cap
        target.update(b_active)
        # 黄金/银行袖珍
        if listed.loc[day, GOLD] or gold_off:
            if gold_off:
                # 黄金失势 → 银行防御(各行自身MA250过滤), B让位
                for b in BANKS:
                    if listed.loc[day, b] and np.isfinite(ma250.loc[day, b]) and c[b] > ma250.loc[day, b]:
                        target[b] = 0.125
                        if b not in pos:
                            trades.append((str(day.date()), "DEF-IN", TICKERS[b], "黄金失势防御"))
            else:
                gw = max(gold_w - b_sum, 0.0)
                if listed.loc[day, GOLD] and GOLD not in oh_ref:
                    target[GOLD] = gw
                    if GOLD not in pos:
                        trades.append((str(day.date()), "CORE-IN", TICKERS[GOLD], f"{gw:.1%}"))
        # 总和归一(银行防御时可能 nasdaq50+banks50+B20=120%)
        tot = sum(target.values())
        if tot > 1.0:
            # 银行防御期: B轨先让位
            excess = tot - 1.0
            for t in list(target):
                if t in B_TRACK and excess > 0:
                    cut = min(target[t], excess)
                    target[t] -= cut
                    excess -= cut
                    if target[t] <= 1e-9:
                        del target[t]
            tot = sum(target.values())
            if tot > 1.0:
                target = {t: w / tot for t, w in target.items()}
        # 碎单过滤
        total_now = cash + mv
        cur_w = {t: n * c.get(t, np.nan) / total_now for t, n in pos.items()
                 if np.isfinite(c.get(t, np.nan))} if total_now > 0 else {}
        changed = (set(target) != set(cur_w)) or \
                  any(abs(target.get(t, 0) - cur_w.get(t, 0)) >= dust for t in set(target) | set(cur_w))
        if changed:
            pending = target

    eq = pd.Series(dict(curve)).sort_index()
    return eq, trades


# ---------------------------------------------------------------------------
# v3: 分资产类别策略 (用户20260806指导)
#   A轨(低波复利): 黄金/纳指/银行  趋势持有直到反转 (MA250 + LLM下降)
#   B轨(高波周期): 恒科/科创/原油  底部反转等待 (深回撤武装→筑底确认→持有到趋势结束)
# ---------------------------------------------------------------------------

def bt_v3(closes, opens, llm_dir=None, cost_etf=0.0005, cost_stock=0.001,
          a_exit_buf=0.02, a_entry_buf=0.01, alloc="normalize", a_ma=250,
          b_arm_dd=-0.25, b_bottom_age=20, b_stop=-0.12,
          b_llm_bottom=False, use_llm_exit=True, use_a=True, use_b=True,
          dust=0.02, b_exit_ma=60):
    """alloc: normalize=超100%按比例归一; priority=按 纳指>黄金>银行>B 优先填满(复利优先)"""
    # 优先fill顺序
    PRIORITY = ["513100.SS", "518800.SS", "601398.SS", "601988.SS", "601939.SS", "601288.SS",
                "513180.SS", "588000.SS", "501018.SS", "160723.SZ"]
    """事件驱动: 每日收盘判定, 次日09:31成交。各票目标权重=类上限(在场), 总和>1按比例归一"""
    days = closes.index.intersection(opens.index)
    listed = closes.notna()
    cshift = closes.shift(1)
    ma250 = closes.rolling(250).mean()
    ma20 = closes.rolling(20).mean()
    ma20_slope = ma20 / ma20.shift(20) - 1
    maX = closes.rolling(b_exit_ma).mean()

    # B轨每票: 250日低点距今交易日数 (预计算)
    low_age = {}
    for t in B_TRACK:
        px = closes[t].values
        age = np.full(len(days), 10**9)
        for i in range(len(days)):
            if not np.isfinite(px[i]):
                continue
            j0 = max(0, i - 249)
            w = px[j0:i + 1]
            if np.isfinite(w).any():
                age[i] = i - (j0 + int(np.nanargmin(w)))
        low_age[t] = age
    hi250 = closes.rolling(250).max()

    # LLM 月度状态 (复用v2 schema: 下降=退出, 筑底=B轨可选确认)
    llm_tl = []
    if llm_dir and os.path.isdir(llm_dir):
        for fn in sorted(os.listdir(llm_dir)):
            if fn.endswith(".json"):
                try:
                    doc = json.load(open(os.path.join(llm_dir, fn), encoding="utf-8"))
                    if doc.get("version") == 2:
                        llm_tl.append((pd.Timestamp(doc["asof"]), doc))
                except Exception:
                    pass
        llm_tl.sort(key=lambda x: x[0])
    llm_state = {"idx": -1, "exit": set(), "bottom": set(), "block": set()}

    cash, pos = 1.0, {}
    b_entry = {}                     # B轨入场价 (硬止损用)
    pending = None
    curve, trades = [], []

    for i, day in enumerate(days):
        c, o, cp = closes.loc[day], opens.loc[day], cshift.loc[day]
        # 0) LLM 状态推进
        while llm_state["idx"] + 1 < len(llm_tl) and llm_tl[llm_state["idx"] + 1][0] <= day:
            llm_state["idx"] += 1
            doc = llm_tl[llm_state["idx"]][1]
            llm_state["exit"] = {t for t, a in doc.get("assets", {}).items()
                                 if a.get("phase") == "下降"}
            llm_state["bottom"] = {t for t, a in doc.get("assets", {}).items()
                                   if a.get("phase") == "筑底"}
            # 月度粘性: 新月报到达时, 仍判下降的维持封锁, 否则解除
            llm_state["block"] = llm_state["block"] & llm_state["exit"]
        # 1) 开盘执行昨日信号
        if pending is not None:
            fee_of = lambda t: cost_stock if t in STOCKS else cost_etf
            px_of = lambda t: o[t] if np.isfinite(o.get(t, np.nan)) else cp.get(t, np.nan)
            total = cash + sum(n * px_of(t) for t, n in pos.items() if np.isfinite(px_of(t)))
            fees, new_pos = 0.0, {}
            for t, n in pos.items():
                px = px_of(t)
                if not np.isfinite(px):
                    new_pos[t] = n
                    continue
                if t in pending:
                    new_pos[t] = n
                else:
                    fees += n * px * fee_of(t)
                    b_entry.pop(t, None)
            for t, w in pending.items():
                px = px_of(t)
                if not np.isfinite(px) or px <= 0:
                    continue
                if t in pos:
                    # 权重调整(归一化引起): 直接按目标重设份额, 差额收一次费
                    cur_mv = pos[t] * px
                    tgt_mv = total * w
                    fees += abs(tgt_mv - cur_mv) * fee_of(t) * 0.5
                    new_pos[t] = tgt_mv / px
                else:
                    new_pos[t] = total * w * (1 - fee_of(t)) / px
                    fees += total * w * fee_of(t)
                    if t in B_TRACK:
                        b_entry[t] = px          # B轨入场价=实际成交(硬止损锚)
            mv_open = sum(n * px_of(t) for t, n in new_pos.items() if np.isfinite(px_of(t)))
            cash = total - mv_open - fees
            pos = new_pos
            pending = None
        # 2) 日终估值
        mv = sum(n * c.get(t, np.nan) for t, n in pos.items() if np.isfinite(c.get(t, np.nan)))
        curve.append((day, cash + mv))
        # 3) 收盘生成目标权重
        target = {}
        if use_a:
            for t, cap in A_TRACK.items():
                if not listed.loc[day, t]:
                    continue
                if a_ma and not np.isfinite(ma250.loc[day, t]):
                    continue
                in_pos = t in pos
                if not in_pos:
                    if t in llm_state["block"]:
                        continue                     # LLM退出封锁: 等下月报
                    if not a_ma or c[t] > ma250.loc[day, t] * (1 + a_entry_buf):
                        target[t] = cap
                        trades.append((str(day.date()), "A-IN", TICKERS[t], f"cap{cap:.1%}"))
                else:
                    rev = a_ma and c[t] < ma250.loc[day, t] * (1 - a_exit_buf)
                    llm_down = use_llm_exit and t in llm_state["exit"]
                    if not rev and not llm_down:
                        target[t] = cap
                    else:
                        if llm_down:
                            llm_state["block"].add(t)
                        trades.append((str(day.date()), "A-OUT", TICKERS[t],
                                       "破MA%d" % a_ma if rev else "LLM下降"))
        if use_b:
            for t, cap in B_TRACK.items():
                if not (listed.loc[day, t] and np.isfinite(ma250.loc[day, t])):
                    continue
                in_pos = t in pos
                if not in_pos:
                    if t in llm_state["block"]:
                        continue                     # LLM退出封锁: 等下月报
                    armed = np.isfinite(hi250.loc[day, t]) and c[t] / hi250.loc[day, t] - 1 <= b_arm_dd
                    trig = (armed and low_age[t][i] >= b_bottom_age
                            and np.isfinite(ma20.loc[day, t]) and c[t] > ma20.loc[day, t]
                            and np.isfinite(ma20_slope.loc[day, t]) and ma20_slope.loc[day, t] > 0
                            and (not b_llm_bottom or t in llm_state["bottom"]))
                    if trig:
                        target[t] = cap
                        trades.append((str(day.date()), "B-IN", TICKERS[t],
                                       f"dd{c[t] / hi250.loc[day, t] - 1:.0%}/低点{low_age[t][i]}日"))
                else:
                    trend_end = np.isfinite(maX.loc[day, t]) and c[t] < maX.loc[day, t]
                    llm_down = use_llm_exit and t in llm_state["exit"]
                    stop = t in b_entry and c[t] / b_entry[t] - 1 <= b_stop
                    if not trend_end and not llm_down and not stop:
                        target[t] = cap
                    else:
                        if llm_down:
                            llm_state["block"].add(t)
                        reason = "趋势结束(破MA%d)" % b_exit_ma if trend_end else ("LLM下降" if llm_down else "硬止损")
                        trades.append((str(day.date()), "B-OUT", TICKERS[t], reason))
        # 组合装配: 超100%时 normalize=按比例 / priority=复利资产优先填满
        if target:
            tot = sum(target.values())
            if tot > 1.0:
                if alloc == "priority":
                    filled, room = {}, 1.0
                    for t in PRIORITY:
                        if t in target and room > 0:
                            filled[t] = min(target[t], room)
                            room -= filled[t]
                    target = filled
                else:
                    target = {t: w / tot for t, w in target.items()}
        cur_w = {}
        total_now = cash + mv
        if total_now > 0:
            cur_w = {t: n * c.get(t, np.nan) / total_now for t, n in pos.items()
                     if np.isfinite(c.get(t, np.nan))}
        changed = (set(target) != set(cur_w)) or \
                  any(abs(target.get(t, 0) - cur_w.get(t, 0)) >= dust for t in set(target) | set(cur_w))
        if changed:
            pending = target

    eq = pd.Series(dict(curve)).sort_index()
    return eq, trades


def metrics(eq, name=""):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(244) if r.std() > 0 else 0
    calmar = ann / abs(mdd) if mdd else 0
    print(f"{name:<28} 总收益 {total:+8.1%}  年化 {ann:+7.2%}  MDD {mdd:7.2%}  Sharpe {sharpe:5.2f}  Calmar {calmar:5.2f}")
    return dict(total=total, ann=ann, mdd=mdd, sharpe=sharpe)


def yearly(eq):
    y = eq.resample("YE").last()
    prev = eq.iloc[0]
    out = {}
    for d, v in y.items():
        out[d.year] = v / prev - 1
        prev = v
    cur = eq.iloc[-1] / prev - 1
    out[eq.index[-1].year] = eq.iloc[-1] / y.iloc[-2] - 1 if len(y) > 1 else cur
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.5])
    ap.add_argument("--topk", type=int, default=0, help="持有只数上限, 0=全部过闸标的(推荐)")
    ap.add_argument("--gate", type=float, default=0.0, help="绝对动量闸门, score<=gate → 现金")
    ap.add_argument("--reb-days", type=int, default=10)
    ap.add_argument("--cost-etf", type=float, default=0.0005, help="ETF单边费率")
    ap.add_argument("--cost-stock", type=float, default=0.001, help="银行股单边费率(含印花税)")
    ap.add_argument("--start", default="2013-08-01", help="回测起点(默认6标的时代)")
    ap.add_argument("--vol-adj", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--exit-daily", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--class-cap", type=int, default=1, help="每资产类最多只数(0=不限)")
    ap.add_argument("--trend-ma", type=int, default=0, help="趋势过滤MA窗口(0=不用)")
    ap.add_argument("--trail-stop", type=float, default=None, help="移动止损(如-0.10)")
    ap.add_argument("--hyst", type=float, default=0.2, help="离场滞后带宽(score单位)")
    ap.add_argument("--cooldown", type=int, default=10, help="风控离场后冷却交易日数")
    ap.add_argument("--inc-margin", type=float, default=0.0, help="在位优势(score单位)")
    ap.add_argument("--weight-mode", choices=["equal", "invvol"], default="invvol")
    ap.add_argument("--max-w", type=float, default=0.35, help="单标的权重上限(0=不限)")
    ap.add_argument("--eq-ma", type=int, default=150, help="净值熔断MA窗口(0=不用)")
    ap.add_argument("--llm-mode",
                    choices=["off", "veto", "scale", "both", "phase", "phase-bottom", "phase-exit", "phase-top", "phase-sell"],
                    default="phase-sell",
                    help="v1证伪档: veto/scale/both; v2阶段档: phase全开/phase-bottom仅早鸟/phase-exit仅退出侧")
    ap.add_argument("--llm-dir", default=os.path.join(os.path.dirname(DATA), "output", "llm_monthly"))
    ap.add_argument("--llm-riskoff-cap", type=float, default=0.5, help="risk_off月仓位乘数(v1)")
    ap.add_argument("--llm-min-conf", type=float, default=0.6, help="LLM判定置信度门槛")
    ap.add_argument("--llm-bottom-w", type=float, default=0.5, help="早鸟票权重上限折扣(逆势半仓)")
    ap.add_argument("--llm-exit-conf", type=float, default=0.0, help="下降退出置信度门槛(0=不过滤)")
    ap.add_argument("--strategy", choices=["v2", "v3", "v31"], default="v2",
                    help="v2=量化轮动+LLM卖出侧; v3=分轨(A趋势持有+B底部反转, 用户20260806)")
    ap.add_argument("--v3-a-exit-buf", type=float, default=0.02, help="A轨破MA250退出缓冲")
    ap.add_argument("--v3-arm-dd", type=float, default=-0.25, help="B轨武装回撤门槛")
    ap.add_argument("--v3-bottom-age", type=int, default=20, help="B轨低点确认交易日数")
    ap.add_argument("--v3-b-stop", type=float, default=-0.12, help="B轨入场硬止损")
    ap.add_argument("--v3-b-exit-ma", type=int, default=60, help="B轨趋势结束MA")
    ap.add_argument("--v3-b-llm-bottom", action="store_true", help="B轨触发需LLM筑底确认")
    ap.add_argument("--v3-no-llm-exit", action="store_true", help="关闭LLM下降退出(消融)")
    ap.add_argument("--v3-no-a", action="store_true", help="关闭A轨(消融)")
    ap.add_argument("--v3-no-b", action="store_true", help="关闭B轨(消融)")
    ap.add_argument("--v3-a-entry-buf", type=float, default=0.01, help="A轨入场缓冲(防抖)")
    ap.add_argument("--v3-alloc", choices=["normalize", "priority"], default="normalize",
                    help="超100%分配: 按比例归一 / 复利优先填满")
    ap.add_argument("--v3-ndx-w", type=float, default=0.50, help="v3.1纳指核心权重")
    ap.add_argument("--v3-gold-w", type=float, default=0.50, help="v3.1黄金核心权重")
    ap.add_argument("--v3-b-total-cap", type=float, default=0.20, help="v3.1 B轨总上限")
    ap.add_argument("--v3-no-oh", action="store_true", help="关闭核心过热止盈(消融)")
    ap.add_argument("--v3-oh-ext", type=float, default=0.275, help="冲顶武装: 偏离MA250阈值")
    ap.add_argument("--v3-use-stag", action="store_true", help="启用滞涨武装(默认关, 假信号多)")
    ap.add_argument("--v3-oh-r20", type=float, default=0.12, help="冲顶武装: ret20阈值")
    ap.add_argument("--v3-stag-ext", type=float, default=0.12, help="滞涨武装: 偏离阈值")
    ap.add_argument("--v3-stag-r250", type=float, default=0.20, help="滞涨武装: ret250阈值")
    ap.add_argument("--v3-stag-r60", type=float, default=0.0, help="滞涨武装: ret60上限")
    ap.add_argument("--v3-oh-cool", type=int, default=20, help="回补后再触发冷却日数")
    ap.add_argument("--v3-oh-re-age", type=int, default=20, help="调整充分回补: 低点确认日数")
    ap.add_argument("--v3-oh-re-slope", action="store_true", help="回补需MA20斜率转正(严格筑底)")
    ap.add_argument("--v3-a-ma", type=int, default=250, help="A轨趋势MA(0=纯持有仅LLM退出)")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    closes, opens = load_daily()
    closes = closes.loc[args.start:]
    opens = opens.loc[args.start:]
    print(f"回测区间: {closes.index[0].date()} -> {closes.index[-1].date()}  ({len(closes)} 交易日)")

    if args.strategy == "v31":
        eq, trades = bt_v31(closes, opens, llm_dir=args.llm_dir,
                            cost_etf=args.cost_etf, cost_stock=args.cost_stock,
                            b_arm_dd=args.v3_arm_dd, b_bottom_age=args.v3_bottom_age,
                            b_stop=args.v3_b_stop, b_exit_ma=args.v3_b_exit_ma,
                            b_total_cap=args.v3_b_total_cap,
                            use_llm_exit=not args.v3_no_llm_exit,
                            ndx_w=args.v3_ndx_w, gold_w=args.v3_gold_w,
                            use_oh=not args.v3_no_oh,
                            oh_ext=args.v3_oh_ext, oh_r20=args.v3_oh_r20,
                            stag_ext=args.v3_stag_ext, stag_r250=args.v3_stag_r250,
                            stag_r60=args.v3_stag_r60, oh_cool=args.v3_oh_cool,
                            use_stag=args.v3_use_stag,
                            oh_re_age=args.v3_oh_re_age, oh_re_slope=args.v3_oh_re_slope)
        metrics(eq, f"v3.1核心{args.v3_ndx_w:.0%}纳指/{args.v3_gold_w:.0%}黄金")
        print("\n== 逐年收益 ==")
        for y, v in yearly(eq).items():
            print(f"  {y}: {v:+.1%}")
        print(f"\n交易次数: {len(trades)}")
        out = os.path.join(os.path.dirname(DATA), "output", "rotation_equity_v31.csv")
        eq.to_csv(out, header=["equity"])
        print("净值已存:", out)
        return
    if args.strategy == "v3":
        eq, trades = bt_v3(closes, opens, llm_dir=args.llm_dir,
                           cost_etf=args.cost_etf, cost_stock=args.cost_stock,
                           a_exit_buf=args.v3_a_exit_buf, b_arm_dd=args.v3_arm_dd,
                           b_bottom_age=args.v3_bottom_age, b_stop=args.v3_b_stop,
                           b_exit_ma=args.v3_b_exit_ma,
                           b_llm_bottom=args.v3_b_llm_bottom,
                           use_llm_exit=not args.v3_no_llm_exit,
                           use_a=not args.v3_no_a, use_b=not args.v3_no_b,
                           a_entry_buf=args.v3_a_entry_buf, alloc=args.v3_alloc,
                           a_ma=args.v3_a_ma)
        metrics(eq, "v3分轨策略")
        # 躺平基准三行
        n = closes["513100.SS"].dropna()
        g = closes["518800.SS"].dropna()
        b4 = closes[["601398.SS", "601988.SS", "601939.SS", "601288.SS"]].dropna()
        bank_eq = b4.div(b4.iloc[0]).mean(axis=1)
        ng = (0.5 * n / n.iloc[0] + 0.5 * g / g.iloc[0]).dropna()
        ngb = (0.4 * n / n.iloc[0] + 0.3 * g / g.iloc[0] + 0.3 * bank_eq).dropna()
        metrics(n / n.iloc[0], "基准: 纯纳指")
        metrics(ng, "基准: 50纳指+50黄金躺平")
        metrics(ngb, "基准: 40纳指+30黄金+30银行躺平")
        print("\n== 逐年收益 ==")
        for y, v in yearly(eq).items():
            print(f"  {y}: {v:+.1%}")
        print(f"\n交易次数: {len(trades)}")
        print("最近 16 笔:")
        for tr in trades[-16:]:
            print("  ", tr)
        out = os.path.join(os.path.dirname(DATA), "output", "rotation_equity_v3.csv")
        eq.to_csv(out, header=["equity"])
        print("净值已存:", out)
        return

    kw = dict(vol_adj=args.vol_adj, exit_daily=args.exit_daily,
              class_cap=args.class_cap, trend_ma=args.trend_ma, trail_stop=args.trail_stop,
              hyst=args.hyst, cooldown=args.cooldown, inc_margin=args.inc_margin,
              weight_mode=args.weight_mode, max_w=args.max_w, eq_ma=args.eq_ma,
              llm_mode=args.llm_mode, llm_dir=args.llm_dir,
              llm_cap=args.llm_riskoff_cap, llm_conf=args.llm_min_conf,
              llm_bottom_w=args.llm_bottom_w, llm_exit_conf=args.llm_exit_conf)
    if args.grid:
        print("\n== 参数网格稳健性 ==")
        for lb, wt in [([20, 60], [0.5, 0.5]), ([10, 30], [0.5, 0.5]),
                       ([60], [1.0]), ([20, 60, 120], [0.25, 0.5, 0.25])]:
            for k in (2, 3):
                for rd in (10, 20):
                    eq, _ = bt(closes, opens, lb, wt, k, args.gate, rd,
                               args.cost_etf, args.cost_stock, **kw)
                    metrics(eq, f"L{lb} K{k} R{rd}")
        return

    eq, trades = bt(closes, opens, args.lookbacks, args.weights, args.topk,
                    args.gate, args.reb_days, args.cost_etf, args.cost_stock, **kw)
    metrics(eq, f"轮动 L{args.lookbacks} K{args.topk}")
    # 基准: 等权买入持有 + 各标的
    bmk = closes.dropna(axis=1, how="all")
    ew = sum(closes[t] / closes[t].dropna().iloc[0] for t in closes.columns) / closes.shape[1]
    metrics(ew.dropna(), "基准: 全池等权买入持有")
    for t in closes.columns:
        s = closes[t].dropna()
        if len(s) > 244:
            metrics(s, f"基准: {TICKERS[t]}")
    print("\n== 逐年收益 ==")
    for y, v in yearly(eq).items():
        print(f"  {y}: {v:+.1%}")
    print(f"\n交易次数: {len(trades)}")
    print("最近 12 笔:")
    for tr in trades[-12:]:
        print("  ", tr)
    os.makedirs(os.path.join(os.path.dirname(DATA), "output"), exist_ok=True)
    out = os.path.join(os.path.dirname(DATA), "output", "rotation_equity.csv")
    eq.to_csv(out, header=["equity"])
    print("净值已存:", out)


if __name__ == "__main__":
    main()
