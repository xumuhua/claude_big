# claude_big · ETF 轮动系统（v3.10）

> 与股票主线策略（claude_stock 仓）平行的**大类资产/行业 ETF 轮动**项目：
> 低波复利资产永持为底仓 + 高波资产底部反转为增强，LLM 月频/事件级文档为护航。
> 决策-执行分离：Linux 每晚全量重跑产指导文件，PTrade 做目标 vs 实际差额下单。

## 一、核心业绩（前复权口径，2013-08-01 → 2026-08-07）

| 指标 | 数值 |
|---|---|
| 总收益 | **+1045.6%**（1元→11.46元） |
| 最大回撤 | **-19.30%** |
| Sharpe / Calmar | **1.43 / 1.07** |
| 交易 | 343 笔 |

- 口径说明（20260807 用户指定）：银行股=common_data 前复权（含股息再投资）；ETF=fund_daily×fund_adj 前复权。修复三个原始价缺陷：518800 份额合并假跳变（2026-04-15 raw 假+6.7%）、513100 拆分、银行股除息假跌（raw 口径年丢 5-6% 股息）、原油 LOF 补齐 2016-2020 历史。
- raw 口径存档成绩 +994.3%@08-04 作废，以复权口径为新基准。

## 二、策略架构（v3.10 定稿）

```
A轨（永持底仓）: 纳指50% + 黄金50%
  ├─ OH冲顶止盈 / 核按钮(-28%/500日窗) / 崩盘梯形接回(-20%/-30%各30%cap)
  └─ 震荡态 RSI 网格
B轨（高波增强）: 恒科/科创50/两油 —— 三层入场(V反早鸟/标准筑底/W2主升浪)
防御替补: 银行4只各12.5% = 黄金失势替补（须在自身MA250上方）
LLM 层: 月频 phase-sell + 事件级衰竭退出/护航
```

## 三、数据链路（本机=数据权威）

```
fetch_stock_price.py::fetch_big_pool()   # 股票仓quant_levels侧, tushare fund_daily×fund_adj
  → ~/local_data/common_data/big_pool_price.csv      # 全量覆写(复权锚随分红漂移)
  → scripts/update_ydata.py                          # 全量重建 y-data.csv(8bar/日倒序)
  → scripts/rotation_backtest.py::run_v31_canonical  # 正典配置单一事实源
```

## 四、实盘部署

```
holding.py（每晚） → scripts/rotation_live.py --run-daily
  update_ydata(容错) → 月末LLM/B轨事件LLM → 全量重跑(天然保真,无孪生漂移)
  → output/live/big_directive_latest.json → 同步 history_output_new/
       ▼
ptrade/ptrade_big.py（market_en 0=回测真实下单 / 1=部署输出hit文件）
  目标仓位驱动: 指导文件=每日全量目标权重快照整体替换; 9:30先卖后买;
  588开头20%涨跌幅其余10%; dust=2%; 整手100份/5万股拆单
  重启容错: big_state.json仅存名称/目标/指导日期(文件缺失沿用昨日目标,绝不误清仓)
```

- 校对包：`output/live/big_directives_pack_20130801_20260807.tar.gz`（3164 份指导文件+README+基准CSV）；回测建议窗口 2025-01-02 起（分钟级）
- 基准：`output/live/big_ref_equity/big_ref_weights_20130801_20260807.csv`
- 当前持仓（08-07 信号）：恒科 15% + 工行/中行/建行各 12.5% + 现金 47.5%

## 五、目录结构

```
~/claude_big/
├── scripts/
│   ├── rotation_backtest.py   # 回测引擎(bt_v31+run_v31_canonical正典入口)
│   ├── rotation_live.py       # 实盘守备(--run-daily/--emit-directives/--check)
│   ├── update_ydata.py        # y-data.csv 全量重建器(复权口径+接缝校验)
│   └── monthly_llm_analysis.py # LLM 月度/事件级分析
├── ptrade/ptrade_big.py       # PTrade 策略(部署/回测双模式)
├── y-data.csv                 # 行情底座(17MB日更,不入git)
└── output/
    ├── llm_monthly/ llm_events/   # LLM 月度/事件文档(决策输入)
    └── live/                      # 指导文件/校对包/日志(big_live_log.jsonl)
```

## 六、快速上手与运维

```bash
# 回测（正典配置）
python3 scripts/rotation_backtest.py --strategy v31
# 每日守备（holding.py 自动调用）
python3 scripts/rotation_live.py --run-daily
# 正典一致性校验
python3 scripts/rotation_live.py --check
# 全历史指导文件重生成（校对包用）
python3 scripts/rotation_live.py --emit-directives
# 数据底座重建
python3 scripts/update_ydata.py
```

**纪律**：优化重启先读已证伪清单（memory quant-etf-rotation-big-20260805）；
改配置必须经 run_v31_canonical 单一事实源（防双默认漂移事故）。
