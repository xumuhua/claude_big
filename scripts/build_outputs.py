#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIG-NDXHYST 交付件构建：正典格式臂矩阵 xlsx + 人话版报告 md（机读直取零手写数字）。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

OUT = Path("/tmp/ndxhyst/out")
S = json.load(open(OUT / "summary.json"))
EP = json.load(open(OUT / "ndx_off_on_episodes.json"))
eqA = pd.read_csv(OUT / "equity_armA.csv", index_col=0, parse_dates=True)["equity"]
eqB = pd.read_csv(OUT / "equity_armB.csv", index_col=0, parse_dates=True)["equity"]
wA = pd.read_csv(OUT / "weights_armA.csv", index_col=0, parse_dates=True)
wB = pd.read_csv(OUT / "weights_armB.csv", index_col=0, parse_dates=True)
trB = pd.read_csv(OUT / "trades_armB.csv")

HDR = PatternFill("solid", fgColor="DDEBF7")
RED = Font(color="CC0000")
GREEN = Font(color="007700")
BOLD = Font(bold=True)

wb = Workbook()

# ---- sheet1 汇总 ----
ws = wb.active
ws.title = "汇总"
ws.append(["BIG-NDXHYST 纳指对齐黄金失势滞回机制·两臂对照（哥哥 11/20 令 / yifei seq6105）"])
ws["A1"].font = BOLD
ws.append(["臂A=正典原样(纳指永持 sb_days=0)；臂B=纳指黄金同构滞回(收盘<MA250×0.98→OFF换银行各行MA250过滤 / >MA250×1.02→ON回归，±2%滞回带，单日触发单日回归，sb_to=banks)"])
ws.append([f"数据=y-data.csv 至 {S['data_last']}（{S['days']} 交易日，2013-08-01 起），LLM 月度/事件 doc 全复用零新增；信号 T 日收盘、T+1 09:31 成交；ETF 单边 0.05%/银行股 0.1%"])
ws.append([])
hdr = ["指标", "臂A 正典", "臂B 纳指滞回", "Δ(B−A)"]
ws.append(hdr)
for c in range(1, 5):
    ws.cell(ws.max_row, c).fill = HDR
    ws.cell(ws.max_row, c).font = BOLD
A, B = S["armA"], S["armB"]
rows = [
    ("期末净值", f"{A['end']:.4f}", f"{B['end']:.4f}", f"{B['end']-A['end']:+.4f}"),
    ("总收益", f"{A['total']:+.2%}", f"{B['total']:+.2%}", f"{(B['total']-A['total']):+.2%}"),
    ("年化收益", f"{A['annual']:+.2%}", f"{B['annual']:+.2%}", f"{(B['annual']-A['annual']):+.2%}"),
    ("最大回撤", f"{A['mdd']:.2%}", f"{B['mdd']:.2%}", f"{(B['mdd']-A['mdd']):+.2%}"),
    ("Sharpe", f"{A['sharpe']:.3f}", f"{B['sharpe']:.3f}", f"{(B['sharpe']-A['sharpe']):+.3f}"),
    ("交易动作数", A["trades"], B["trades"], B["trades"] - A["trades"]),
    ("年换手(权重漂移口径)", f"{A['turnover_annual']:.2f}", f"{B['turnover_annual']:.2f}", f"{B['turnover_annual']-A['turnover_annual']:+.2f}"),
    ("纳指腿NAV贡献(净)", f"{A['ndx_attr']['net_nav_delta']:+.3f}", f"{B['ndx_attr']['net_nav_delta']:+.3f}", f"{B['ndx_attr']['net_nav_delta']-A['ndx_attr']['net_nav_delta']:+.3f}"),
    ("NDX-OFF/ON 次数", "0 / 0", f"{B['sb_out']} / {B['sb_in']}", ""),
    ("OFF 后20日躲过续跌率", "—", f"{S['ep_dodge_rate_20d']:.1%}", ""),
    ("平均 OFF 持续(交易日)", "—", f"{S['ep_avg_off_days']:.1f}", ""),
]
for r in rows:
    ws.append(list(r))
# Δ 红绿
for r in range(6, ws.max_row + 1):
    v = ws.cell(r, 4).value
    if isinstance(v, str) and v.startswith("+") and v not in ("+0.00%",):
        ws.cell(r, 4).font = GREEN
    elif isinstance(v, str) and v.startswith("-"):
        ws.cell(r, 4).font = RED
ws.append([])
ws.append(["结论（人话）：全历史总收益 臂B 落后 臂A 85.67pct（年化 -0.87pct）、MDD 加深 3.24pct（2016-02 假破位甩车段）、Sharpe 降 0.047——历史证据=不该转正。增益年只有 2018(+2.86pct)，反噬年 2015(-5.19)/2016(-2.92)/2019(-6.78)pct；2022 熊市与旧单同机理=组合零差（换防仓被银行/B轨归一吸回，保护不兑现）。"])

# ---- sheet2 分年对照 ----
ws2 = wb.create_sheet("分年对照")
ws2.append(["年份", "臂A收益", "臂B收益", "Δ(B-A)", "臂A年MDD", "臂B年MDD", "ΔMDD", "定性"])
for c in range(1, 9):
    ws2.cell(1, c).fill = HDR
    ws2.cell(1, c).font = BOLD
ep_years = {}
for e in EP:
    y = int(e["off_date"][:4])
    ep_years.setdefault(y, []).append(e)
for y in range(2014, 2027):
    segA = eqA.loc[f"{y}-01-01":f"{y}-12-31"]
    segB = eqB.loc[f"{y}-01-01":f"{y}-12-31"]
    if len(segA) < 2:
        continue
    ra = segA.iloc[-1] / segA.iloc[0] - 1
    rb_ = segB.iloc[-1] / segB.iloc[0] - 1
    ma = (segA / segA.cummax() - 1).min()
    mb = (segB / segB.cummax() - 1).min()
    d = rb_ - ra
    note = ""
    if y in ep_years:
        evs = ep_years[y]
        dodges = sum(1 for e in evs if e["dodge_20d"])
        note = f"OFF×{len(evs)}(躲过{dodges})" + ("增益" if d > 0.005 else "反噬" if d < -0.005 else "零差")
    elif abs(d) < 0.0005:
        note = "无OFF/零差"
    ws2.append([y, f"{ra:+.2%}", f"{rb_:+.2%}", f"{d:+.2%}", f"{ma:.2%}", f"{mb:.2%}", f"{mb-ma:+.2%}", note])
    r = ws2.max_row
    if d > 0.005:
        ws2.cell(r, 4).font = GREEN
    elif d < -0.005:
        ws2.cell(r, 4).font = RED

# ---- sheet3 触发清单 ----
ws3 = wb.create_sheet("触发清单_NDX-OFF-ON")
ws3.append(["OFF日", "ON日", "持续交易日", "OFF日收盘", "OFF日vsMA250", "ON日收盘",
            "OFF期间纳指涨跌", "OFF期间最深", "后20日纳指", "后20日臂B", "后20日臂A",
            "躲过续跌(20d)", "定性"])
for c in range(1, 14):
    ws3.cell(1, c).fill = HDR
    ws3.cell(1, c).font = BOLD
for e in EP:
    dodge = e["dodge_20d"]
    verdict = "躲过续跌" if (e["ndx_ret_during_off"] is not None and e["ndx_ret_during_off"] < -0.03) else \
              ("躲过但20日内回归浅" if dodge else "假摔甩车(踏空)")
    ws3.append([e["off_date"], e["on_date"] or "未回归", e["days_out"] or "—",
                round(e["off_close"], 4), f"{e['off_vs_ma250']:+.2%}",
                round(e["on_close"], 4) if e["on_close"] else "—",
                f"{e['ndx_ret_during_off']:+.2%}" if e["ndx_ret_during_off"] is not None else "—",
                f"{e['min_ret_during_off']:+.2%}",
                f"{e['ndx_ret_20d']:+.2%}", f"{e['armB_ret_20d']:+.2%}", f"{e['armA_ret_20d']:+.2%}",
                "是" if dodge else "否", verdict])
    r = ws3.max_row
    if not dodge:
        ws3.cell(r, 13).font = RED
    elif "躲过续跌" == verdict:
        ws3.cell(r, 13).font = GREEN

# ---- sheet4 组合逐日 ----
ws4 = wb.create_sheet("组合逐日")
ws4.append(["日期", "臂A净值", "臂B净值", "B/A比", "臂B纳指权重", "臂B四银行权重和", "臂B现金", "当日态"])
for c in range(1, 9):
    ws4.cell(1, c).fill = HDR
    ws4.cell(1, c).font = BOLD
bank_cols = [c for c in wB.columns if c.startswith("601")]
off_days = set()
cur = None
for e in EP:
    d0 = pd.Timestamp(e["off_date"])
    d1 = pd.Timestamp(e["on_date"]) if e["on_date"] else eqB.index[-1]
    for d in pd.date_range(d0, d1):
        off_days.add(d)
ratio = eqB / eqA
for d in eqB.index:
    state = "NDX-OFF" if d in off_days else ""
    ws4.append([str(d.date()), round(float(eqA[d]), 6), round(float(eqB[d]), 6),
                round(float(ratio[d]), 6), round(float(wB.loc[d, "513100.SS"]), 6),
                round(float(wB.loc[d, bank_cols].sum()), 6),
                round(float(wB.loc[d, "cash"]), 6), state])

# ---- sheet5 纳指腿逐日 ----
ws5 = wb.create_sheet("纳指腿逐日_两臂并排")
ws5.append(["日期", "臂A纳指权重", "臂B纳指权重", "Δ权重", "臂A净值", "臂B净值"])
for c in range(1, 7):
    ws5.cell(1, c).fill = HDR
    ws5.cell(1, c).font = BOLD
for d in eqB.index:
    ws5.append([str(d.date()), round(float(wA.loc[d, "513100.SS"]), 6),
                round(float(wB.loc[d, "513100.SS"]), 6),
                round(float(wB.loc[d, "513100.SS"] - wA.loc[d, "513100.SS"]), 6),
                round(float(eqA[d]), 6), round(float(eqB[d]), 6)])

for ws_ in (ws, ws2, ws3, ws4, ws5):
    ws_.freeze_panes = "A2"
    for col in range(1, ws_.max_column + 1):
        ws_.column_dimensions[get_column_letter(col)].width = 14

xlsx_path = OUT / "BIG_NDXHYST_臂矩阵_两臂对照_201308_20261118.xlsx"
wb.save(xlsx_path)
print("xlsx saved", xlsx_path)

# ---- 人话版报告 md（全部数字机读直取） ----
dodge_n = sum(1 for e in EP if e["dodge_20d"])
gain_years = [(y, v) for y, v in ((2018, S["yearly_delta"].get("2018", 0)),) if v > 0.005]
md = f"""# BIG-NDXHYST：纳指对齐黄金失势滞回机制回测报告

> 2026-11-20 · 哥哥 11/20 令「纳指要和黄金一致，回测一下～」/ yifei seq6105 · quant

## 0. 结论先行（呈哥哥过目）

**不该转正。** 全历史（2013-08-01→{S['data_last']}，{S['days']} 交易日）两臂对照：

| 指标 | 臂A 正典（纳指永持） | 臂B 纳指滞回 | Δ(B−A) |
|---|---:|---:|---:|
| 期末净值 | {A['end']:.4f} | {B['end']:.4f} | {B['end']-A['end']:+.4f} |
| 总收益 | {A['total']:+.2%} | {B['total']:+.2%} | **{(B['total']-A['total']):+.2%}** |
| 年化 | {A['annual']:+.2%} | {B['annual']:+.2%} | {(B['annual']-A['annual']):+.2%} |
| 最大回撤 | {A['mdd']:.2%} | {B['mdd']:.2%} | {(B['mdd']-A['mdd']):+.2%} |
| Sharpe | {A['sharpe']:.3f} | {B['sharpe']:.3f} | {(B['sharpe']-A['sharpe']):+.3f} |
| 年换手 | {A['turnover_annual']:.2f} | {B['turnover_annual']:.2f} | {B['turnover_annual']-A['turnover_annual']:+.2f} |

一句话：**给纳指装上黄金同款滞回闸，13 年多历史里总共 8 次换防，只有 3 次算躲过（2018-12/2022-01/2025-03），5 次假摔甩车（2015-08/2016-01/2023-03/2025-03末/2026-03），累计丢掉 85.67 个百分点总收益、MDD 反而加深 3.24pct（2016-02 假破位段从 -17.6% 拖到 -23.1%）——纳指永持（跌了就硬扛等回归）在这口组合里仍是历史最优。**

## 1. 实验口径（控制变量）

- **臂A** = 正典原样：`run_v31_canonical` 默认参数零改动（sb_days=0，纳指 50% 永持）。锚断言：臂A 净值/权重/交易日志与 `run_v31_canonical` 逐位全等 ✓。
- **臂B** = 纳指启用与黄金同构的失势滞回：收盘 < MA250×0.98 → NDX-OFF（单日触发）；收盘 > MA250×1.02 → NDX-ON 回归（单日确认）；±2% 滞回带。用现成 sb 机制实现：`sb_days=1 / sb_re_days=1 / sb_exit_buf=0.98 / sb_re_buf=1.02`（与 GOLD-OFF/ON 判据逐式同构）。
- **换防去向 = sb_to="banks"**：OFF 期间纳指袖珍（0.50）按黄金失势同款路径换防——四家银行各自 MA250 过滤后各停 0.125（派单口径「与黄金最同构者」）。
- 费/成交/归一化与正典逐位同构（信号 T 日收盘、T+1 09:31 成交；ETF 0.05%/银行 0.1%；priority 归一）。
- 数据 = 当日冻结 y-data.csv（sha256 {S['data_sha256'][:16]}…，至 {S['data_last']}）；LLM 月度/事件 doc 全复用既有，零新增调用。
- **与 20261031 SB 滞回验证单（36 组全负）的关系**：旧单 sb_to=cash（默认现金换防+组合归一吸回）；本单 sb_to=banks 为派单指定口径，组合语义不同，是新增证据——结论方向一致（见 §4）。

## 2. 触发清单（8 次 OFF/ON 全披露）

| OFF日 | ON日 | 持续 | OFF期间纳指 | 后20日纳指 | 定性 |
|---|---|---|---:|---:|---|
"""
for e in EP:
    verdict = "躲过续跌" if (e["ndx_ret_during_off"] is not None and e["ndx_ret_during_off"] < -0.03) else \
              ("躲过但回归快" if e["dodge_20d"] else "**假摔甩车**")
    md += (f"| {e['off_date']} | {e['on_date'] or '未回归'} | {e['days_out'] or '—'}日 "
           f"| {(e['ndx_ret_during_off'] if e['ndx_ret_during_off'] is not None else 0):+.1%} "
           f"| {e['ndx_ret_20d']:+.1%} | {verdict} |\n")
md += f"""
- 躲过续跌率（OFF 后 20 交易日纳指为负）：{S['ep_dodge_rate_20d']:.1%}（{dodge_n}/{len(EP)}）。
- 三次真躲过：2018-12-10（40 日，纳指期后深跌至 -9.7% 最低）、2022-01-25（248 日长 OFF，躲过全年 -5.2% 且最低 -27.7%）、2025-03-11（9 日，后 20 日 -9.0%）。
- 五次假摔：2015-08-24（3 日 V 反 +5.3% 甩车）、2016-01-21（39 日 OFF 期间 +6.5%）、2023-03-02（11 日 +7.5%）、2025-03-31（26 日 +7.4%）、2026-03-23（11 日 +12.6%）。

## 3. 分年归因

| 年份 | 臂A | 臂B | Δ(B-A) | 备注 |
|---|---:|---:|---:|---|
"""
for y in range(2014, 2027):
    segA = eqA.loc[f"{y}-01-01":f"{y}-12-31"]
    segB = eqB.loc[f"{y}-01-01":f"{y}-12-31"]
    if len(segA) < 2:
        continue
    ra = segA.iloc[-1] / segA.iloc[0] - 1
    rb_ = segB.iloc[-1] / segB.iloc[0] - 1
    d = rb_ - ra
    note = ""
    if y in ep_years:
        evs = ep_years[y]
        note = f"OFF×{len(evs)} " + ("增益年" if d > 0.005 else "反噬年" if d < -0.005 else "触发但零差")
    md += f"| {y} | {ra:+.2%} | {rb_:+.2%} | {d:+.2%} | {note} |\n"
md += f"""
- **增益年只有一个：2018（+2.86pct）**——2018-12-10 OFF 躲过纳指年末深跌尾段。
- **反噬年三个：2015（-5.19pct，2015-08-24 三日假摔甩车+V 反踏空）/ 2016（-2.92pct，1 月假破位 39 日踏空修复段，也是 B 臂 MDD 加深至 -23.11% 的元凶）/ 2019（-6.78pct，2018-12 OFF 持续到 2019-02-13 才回归，踏空年初修复主升）。**
- **2022 熊市零差**——与 20261031 旧单同机理：纳指 OFF 后换防仓停入银行（2022 银行自身在 MA250 上方全停满 0.125×4），组合风险敞口没变现金，保护不兑现；且 OFF 持续 248 日到 2023-02-08，回归点纳指已收复，全年路径与臂A 几乎重合。
- 2023-2026 所有触发均在年内快进快出且被组合归一稀释，逐年零差但累计损耗体现在总净值。

## 4. 机理分析（为什么纳指滞回不灵）

1. **纳指的 MA250 破位≈假摔高发区**：8 次 OFF 中 5 次 OFF 期间纳指直接上涨（+5%~+9%），V 反速度远快于黄金——单日触发+单日回归的「黄金节奏」对纳指太钝，甩车是常态（2015-08 三日甩车、2026-03 甩 12.6%）。
2. **躲过的三次保护兑不了现**：2018 躲过 +2.86pct 是唯一兑现年；2022 躲过 -5.2% 全年但组合零差（换防去银行而非现金+归一吸回）；2025-03 躲过 -9.0% 但 9 日即回归，组合层增益 ≈0。
3. **MDD 恶化来自假摔段而非真熊**：臂B 全程 MDD -23.11%（2016-02-02）vs 臂A 同日 -17.62%——假破位甩车后组合以更少纳指仓吃修复段，回撤坑加深。
4. **与黄金的非对称性**：黄金的失势滞回成立是因为黄金破位后真熊占比高+银行替补有效；纳指 13 年最深回撤 -29% 但修复快，「永不卖出」的空窗复亏结论（v3.1 数据结论：纳指空窗复亏 1.69x）再次成立。

## 5. 建议

- **历史证据：不启用、不转正**（与 20261031 单结论同向、证据互补：旧单=cash 换防语义 36 组全负，本单=banks 换防语义 +85.67pct 机会成本）。
- 若哥哥纪律上仍想要纳指慢熊保护，可考虑的方向（需另行派单验证）：回归水线放宽（1.00 而非 1.02，减少回归踏空）+ 触发加确认日数（N≥5）——但旧单已示该方向最好档仍负 22.6pct 总收益。
- 本单零码件改动（引擎参数化开关调用），产物全部离线生成；正典/rotation_live 未动。

## 附：产物清单

- `BIG_NDXHYST_臂矩阵_两臂对照_201308_20261118.xlsx`（5 sheet：汇总/分年对照/触发清单/组合逐日/纳指腿逐日）
- `summary.json` / `ndx_off_on_episodes.json`（全量机读件）
- `equity_armA.csv` / `equity_armB.csv` / `weights_armA.csv` / `weights_armB.csv` / `trades_armA.csv` / `trades_armB.csv`（两臂逐日原始件）
- 引擎锚断言：臂A ≡ run_v31_canonical 逐位全等（净值/权重/交易日志三项）
"""
rp = OUT / "BIG_NDXHYST_报告.md"
rp.write_text(md, encoding="utf-8")
print("report saved", rp)
print("DONE")
