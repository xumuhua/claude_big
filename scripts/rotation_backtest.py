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
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    closes, opens = load_daily()
    closes = closes.loc[args.start:]
    opens = opens.loc[args.start:]
    print(f"回测区间: {closes.index[0].date()} -> {closes.index[-1].date()}  ({len(closes)} 交易日)")

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
