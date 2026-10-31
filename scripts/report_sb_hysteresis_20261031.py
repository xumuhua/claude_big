#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总冻结正典与敏感性结果；在分析脚本落盘后运行。"""
import json
import hashlib
from pathlib import Path

ROOT = Path("/data/workspace/quant/sb_hysteresis_20261031")


def pct(x):
    return f"{100*x:+.2f}%"


def main():
    r = json.loads((ROOT / "sensitivity_enriched.json").read_text())
    base, best, worst = r[0], max(r[1:], key=lambda x: x["total"]), min(r[1:], key=lambda x: x["total"])
    conservative = next(x for x in r if x["label"] == "e0.97_r1.00_n10_m10")
    a = json.loads((ROOT / "analysis.json").read_text())
    bench = a["benchmarks"]
    diag = json.loads((ROOT / "diagnostic_priority.json").read_text())
    priority_diag = diag[1]
    lines = []
    best_eps = a["episodes"][best["label"]]
    conservative_eps = a["episodes"][conservative["label"]]
    best_y = a["yearly_returns"][best["label"]]
    base_y = a["yearly_returns"]["off"]
    lines += ["# ETF-A轨 纳指慢性破位换防 + 水线滞回验证报告", "",
              "> 2026-10-31 · 哥哥直令 / yifei seq4456 · quant", "",
              "**研究状态：回测已完成；正典未启用（sb_days仍为0），rotation_live未动。**", "",
              "## 0. 人话版结论（呈哥哥过目）", "",
              f"1. **双水线已做成，但36组参数没有一组跑赢现有正典。** 现配置关着SB：总收益 {pct(base['total'])}、MDD {pct(base['mdd'])}、Sharpe {base['sharpe']:.3f}；最好的一组只有 {pct(best['total'])} / {pct(best['mdd'])} / {best['sharpe']:.3f}。",
              f"2. **慢N+M能少洗，但代价不小。** 全网格SB退出2–6次，研究口径假破位0–3次；`0.97/1.00/10/10`严格档只退出{conservative['sb_out']}次、假破位0次。代价是2018年底、2022年初和2025年V反中的恢复段踏空。纳指腿贡献从 {base['ndx_attribution']['net_nav_delta']:.3f} NAV降到最好档 {best['ndx_attribution']['net_nav_delta']:.3f}，少 {best['ndx_attribution']['net_nav_delta']-base['ndx_attribution']['net_nav_delta']:.3f} NAV。",
              "3. **2022没有拿到保护。** 所有36档2022年组合收益与基线完全相同：-12.43%（首日归一口径）。原因是纳指换防后，黄金失势替补银行/防御期B轨优先把空出的仓位填回，组合风险并未按字面换成现金。",
              f"4. **把B轨在换防期让位也试过，仍失败。** 同最好参数、让B让位：总收益 {pct(priority_diag['total'])}、MDD {pct(priority_diag['mdd'])}（比不让位还低、回撤还深）。所以不能只靠换替补优先级救活。",
              "5. **我的技术建议：按历史证据不启用；若哥哥拍板要启用，选严格少触发档而非短M收益最高档。** 严格档 = 出场0.97、回归1.00、N=10、M=10；样本内总收益" + pct(conservative['total']) + "，并非未来收益保证。收益最高的是N=3/M=3（" + pct(best['total']) + "），但6次退出中3次为研究口径假破位，且M=3在回归1.00档可跨带累计，不能把它包装成『最防抖』。可以先挂观察/影子，不接实盘；是否启用及何时进rotation_live均等哥哥令。",
              "", f"**直接给哥哥的决策句：**这套水线能少洗几次，但在当前组合里换防不是真空仓，熊市保护没兑现，反而丢复利。历史最优仍是不开；若纪律上一定要开，建议只认0.97/1.00/10日/10日这档，并接受约{-conservative['total_vs_base_pct']:.2f}个百分点累计收益的历史机会成本。",
              "", "## 1. 研究边界与正典一致性", "",
              "- 数据：2013-08-01→2026-10-30，3,244个交易日，冻结 y-data.csv SHA256 `f83f6862679267b63896a64deff8349a71f3df41d2ea2bc8e66d618214461785`；月度/事件LLM文档也冻结，清单在 `input_manifest.json`。",
              "- 引擎：同一 `bt_v31` / `run_v31_canonical`；信号当日收盘、次日09:31成交，ETF单边0.05%、股票单边0.1%。除SB开关/水线/N/M和明示诊断项外参数不变。",
              f"- 基线逐项对齐：终值 {base['end']:.6f}，{base['trade_actions']}条交易动作；默认新参数不改变正典。旧版SB（N=M=5，旧0.98/1.00）与HEAD原版净值/权重/交易日志逐项一致。",
              "- 回归另测：三日慢破后，一天过独立回归线、次日回到中间带，不得凑满M日回归；宽回归线跨出场线仍可正常回归。测试脚本 `scripts/verify_sb_hysteresis_20261031.py`。",
              "- 旧档：20260806曾证伪20–30日SB；本次是哥哥新指定短N/M+滞回带的重新验证，不沿用旧结论代替新证据。",
              "", "## 2. 开关对照 + 躺平基准", "",
              "| 方案 | 总收益 | MDD | Sharpe | 交易动作 | SB退出 | 研究口径假破位 | 2018 | 2022 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, name in [("off", "正典：SB关闭"), (best["label"], f"收益最高档：0.97/1.00/N{best['sb_days']}/M{best['sb_re_days']}"),
                        (conservative["label"], "严格少触发档：0.97/1.00/N10/M10"),
                        (worst["label"], f"最差档：{worst['exit_buf']:.2f}/{worst['re_buf']:.2f}/N{worst['sb_days']}/M{worst['sb_re_days']}")]:
        x = next(z for z in r if z["label"] == label)
        lines.append(f"| {name} | {pct(x['total'])} | {pct(x['mdd'])} | {x['sharpe']:.3f} | {x['trade_actions']} | {x['sb_out']} | {x['false_breakouts']} | {pct(x['y2018']['total'])} | {pct(x['y2022']['total'])} |")
    lines += [f"| 50%纳指+50%黄金买入持有 | {pct(bench['passive_50ndx_50gold']['total'])} | {pct(bench['passive_50ndx_50gold']['mdd'])} | {bench['passive_50ndx_50gold']['sharpe']:.3f} | — | — | — | — | — |", "",
              f"交易动作是引擎规则事件数，不是PTrade成交笔数；每个SB退出+回归构成一个完整往返。收益最高档多{best['trade_actions']-base['trade_actions']}条动作，含{best['sb_out']}次退出、{best['sb_in']}次回归；严格档多{conservative['trade_actions']-base['trade_actions']}条动作。",
              "", "## 3. 36档敏感性参数表", "",
              "所有数字同冻结输入、同引擎；`Δ总`和`ΔMDD`均为相对基线的百分点。完整逐档净值、交易明细与归因在数据目录。",
              "", "| 出场×MA250 | 回归×MA250 | N | M | 总收益 | MDD | Sharpe | Δ总 | ΔMDD | 动作 | SB出 | 假破位 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in sorted(r[1:], key=lambda z: (z["exit_buf"], z["re_buf"], z["sb_days"], z["sb_re_days"])):
        lines.append(f"| {x['exit_buf']:.2f} | {x['re_buf']:.2f} | {x['sb_days']} | {x['sb_re_days']} | {pct(x['total'])} | {pct(x['mdd'])} | {x['sharpe']:.3f} | {x['total_vs_base_pct']:+.2f} | {x['mdd_vs_base_pct']:+.2f} | {x['trade_actions']} | {x['sb_out']} | {x['false_breakouts']} |")
    lines += ["", "**参数方向：**回归带从1.00升到1.02让回归更慢、收益再降；N=3/10优于N=5的平台关系主要是2025是否躲过V反，不能外推成普遍规律。没有形成满足『收益不降+MDD不加深+Sharpe改善』的采纳平台。",
              "", "## 4. 熊市响应与组合装配（关键发现）", "",
              "| 场景 | 最好档行为 | 2018/2022读数 |",
              "|---|---|---|",
              f"| 2018熊市后段 | {best_eps[0]['out']}首次SB-OUT；距前250日最高收盘{best_eps[0]['signal_lag_td']}交易日，出场时已较峰回落{-best_eps[0]['drawdown_from_prior_peak']:.2%}；{best_eps[0]['in']}回归 | 2018全年{pct(best['y2018']['total'])} vs 基线{pct(base['y2018']['total'])}，改善{(best['y2018']['total']-base['y2018']['total'])*100:.2f}pct；2019年{100*best_y['2019']:.2f}% vs {100*base_y['2019']:.2f}%，明显踏空恢复段 |",
              f"| 2022第一轮 | {best_eps[1]['out']}退出，{best_eps[1]['in']}回归；纳指同期涨{best_eps[1]['price_return_to_in']:.2%} | 2022全年仍-12.43%，与基线完全相同 |",
              f"| 2022后续两轮 | {best_eps[2]['out']}→{best_eps[2]['in']}、{best_eps[3]['out']}→{best_eps[3]['in']}；分别退出86/105交易日 | 纳指局部有规避，但其他防御/B轨装配抵消；2022全年仍不变 |",
              "", f"**为什么『转cash』≠『组合持现金』：**现有引擎的A轨黄金失势会配银行，且防御期B轨优先于银行让位。纳指50%退出后，其他槽位照常补进，组合仍是风险资产。补充诊断把换防期改为B先让位：总收益{pct(priority_diag['total'])}、MDD{pct(priority_diag['mdd'])}，也不成立。",
              "", "## 5. 假破位与V反案例", "",
              "**定义（报告统一口径）：**出场时满足现有震荡闸门（距MA250≤8%且120日动量绝对值≤10%）、60交易日内回归、退出后最大再跌不足10%，记为『研究口径假破位』。这是描述性标签，不是因果估计；完整案例见 `episodes.json`。",
              "", "| 参数档 | 退出→回归 | 纳指退出后→回归 | 类型与结果 |",
              "|---|---|---:|---|",
              f"| 0.97/1.00/N3/M3 | {best_eps[0]['out']}→{best_eps[0]['in']} | {best_eps[0]['price_return_to_in']:+.2%} | 震荡区假破位；少参与恢复 |",
              f"| 0.97/1.00/N3/M3 | {best_eps[1]['out']}→{best_eps[1]['in']} | {best_eps[1]['price_return_to_in']:+.2%} | 仅4个交易日回归；震荡区假破位，熊市反弹被错过 |",
              f"| 0.97/1.00/N3/M3 | {best_eps[2]['out']}→{best_eps[2]['in']} | {best_eps[2]['price_return_to_in']:+.2%} | 震荡区假破位，跨带凑满短M回归 |",
              f"| 0.97/1.00/N10/M10 | {conservative_eps[0]['out']}→{conservative_eps[0]['in']} | {conservative_eps[0]['price_return_to_in']:+.2%} | 2018底退出仍错过部分恢复；组合2018全年不变 |",
              f"| 0.97/1.00/N10/M10 | {conservative_eps[1]['out']}→{conservative_eps[1]['in']} | {conservative_eps[1]['price_return_to_in']:+.2%} | 2022反弹段仍有机会成本 |",
              f"| 0.97/1.00/N10/M10 | {conservative_eps[2]['out']}→{conservative_eps[2]['in']} | {conservative_eps[2]['price_return_to_in']:+.2%} | 长换防纳指局部躲过；组合2022全年不变 |",
              "| 0.97/1.00/N3/M3 | 2025-04-08→04-29 | +9.63% | 快N在V反前后洗出；N=5/M=5档恢复更晚 |",
              "", "## 6. 纳指腿单边贡献（口径透明）", "",
              f"基线：毛贡献 {base['ndx_attribution']['gross_nav_delta']:.4f} NAV、估计换手费 {base['ndx_attribution']['estimated_fee']:.4f}、净贡献 {base['ndx_attribution']['net_nav_delta']:.4f}。",
              f"最好档：毛贡献 {best['ndx_attribution']['gross_nav_delta']:.4f}、估计换手费 {best['ndx_attribution']['estimated_fee']:.4f}、净贡献 {best['ndx_attribution']['net_nav_delta']:.4f}；相对基线净贡献 {best['ndx_attribution']['net_nav_delta']-base['ndx_attribution']['net_nav_delta']:+.4f} NAV。",
              "算法：上日实际收盘权重×当日纳指涨跌，扣纳指权重变动×ETF单边费；是按当前组合日度权重拆出的贡献，不等同于独立纳指策略收益，也不含资金再配置的交叉机会成本。",
              "", "## 7. 启用建议与未做事项", "",
              "### 技术结论：历史验证不通过，默认不建议启用", "",
              "36档全部收益/Sharpe下降、MDD不改善；2022组合保护为零。水线改造本身实现正确，但当前正典的装配机制抵消了『换防降险』，不能仅靠水线参数上线。",
              "", "### 若哥哥仍拍板启用：给一档而非模糊网格", "",
              "- 出场：MA250×0.97连续10日；回归：站上MA250×1.00连续10日；换防目标cash；其余正典（含银行/B轨装配）不动。",
              f"- 历史结果：{pct(conservative['total'])} / MDD{pct(conservative['mdd'])} / Sharpe{conservative['sharpe']:.3f} / {conservative['trade_actions']}动作 / {conservative['sb_out']}次SB退出；相对现正典总收益{conservative['total_vs_base_pct']:+.2f}pct。这是按『少洗出』目标挑的严格档，不是收益最优，仍需接受机会成本。",
              f"- 对照：若只看收益最高，N=3/M=3为{pct(best['total'])}，但有{best['sb_out']}次退出/{best['false_breakouts']}次研究口径假破位，不符合首选『防抖』的方向。",
              "- 建议只先挂观察/影子，不直接接实盘；若要求实盘保护，应另案明确换防期是否让银行/B轨全部让位，再重验，而不是把本报告当成上线许可。",
              "- 截至2026-10-30纳指收盘2.239、MA250约2.03494，价在MA250上方约10.03%；离0.98/0.97出场水线均尚远，上线或不上线都不改变当日持仓判断。",
              "", "**未做：**未改正典sb_days（仍0）；未改rotation_live；未发ptrade/平台文件；未承诺样本外有效；未扩大额外指标/全局参数搜索。",
              "", "## 8. 产物与复算", "",
              "- 引擎新增可选参数：`sb_exit_buf` / `sb_re_buf`；正典默认关闭状态不变。",
              "- 脚本：`run_sb_hysteresis_20261031.py`（网格）、`analyze_sb_hysteresis_20261031.py`（独立复算）、`diagnose_sb_priority_20261031.py`（装配隔离）、`verify_sb_hysteresis_20261031.py`（回归）。",
              "- 数据：`/data/workspace/quant/sb_hysteresis_20261031/`：37档净值/逐档交易明细、sensitivity.json/csv、episodes/events、input_manifest、analysis.json。",
              "- runner：`sb-hysteresis-delivery-20261031.service` 完成37/37；`sb-hysteresis-analysis-delivery-20261031.service` 独立复算；`sb-priority-delivery-20261031.service` 二级诊断。",
              "", "—— quant", ""]
    (ROOT / "report.md").write_text("\n".join(lines))
    digest = hashlib.sha256((ROOT / "sensitivity.json").read_bytes()).hexdigest()
    summary = {"cases": len(r), "sensitivity_sha256": digest,
               "baseline": {k: base[k] for k in ["total", "mdd", "sharpe", "trade_actions"]},
               "best_total": {k: best[k] for k in ["label", "total", "mdd", "sharpe", "sb_out", "false_breakouts"]},
               "conservative": {k: conservative[k] for k in ["label", "total", "mdd", "sharpe", "sb_out", "false_breakouts", "total_vs_base_pct"]},
               "all_worse_total": all(x["total"] < base["total"] for x in r[1:]),
               "all_no_better_mdd": all(x["mdd"] <= base["mdd"] + 1e-10 for x in r),
               "y2022_all_identical": len({round(x["y2022"]["total"], 10) for x in r}) == 1,
               "canonical_still_off": True}
    (ROOT / "report_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("REPORT_WRITTEN", ROOT / "report.md", len(lines), digest)


if __name__ == "__main__":
    main()
