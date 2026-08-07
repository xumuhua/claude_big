# claude_big ETF轮动 v3.10 · PTrade 回测校对包（20260807，复权口径）

## 内容

| 文件 | 说明 |
|---|---|
| `directives/big_directive_YYYYMMDD.json` × 3164 | 每日指导文件（信号日收盘装配的目标权重，exec_date=次交易日开盘执行） |
| `big_ref_equity_20130801_20260807.csv` | 基准净值（bt_v31 正典，T+1 09:31 开盘成交口径，含成本） |
| `big_ref_weights_20130801_20260807.csv` | 基准每日实际持仓权重 + 净值（对账主参照） |
| `ptrade_big.py` | PTrade 策略本体（与 `~/claude_big/ptrade/` 一致） |

## 基准成绩（2013-08-01 → 2026-08-07，前复权口径）

**+1045.6% / MDD -19.30% / Sharpe 1.43 / Calmar 1.07**，交易 343 笔

**口径变更（20260807 用户指定）**：银行股=common_data 前复权（含股息再投资，
raw 口径每年白丢 5-6% 股息）；ETF=fund_daily×fund_adj 前复权（修复 518800
2026-04-15 份额合并的假 +6.7% 跳变/513100 拆分由因子原生覆盖；原油 LOF 补齐
2016-2020 历史 → 交易笔数 194→343）。raw 口径存档成绩 +994.3%@08-04 作废，
以复权口径为新基准。逐年（复权 vs raw）：2021 +19.0% vs +5.8%（原油历史补齐）、
2026 +23.5% vs +31.5%（黄金假跳变挤水）、其余年份 ±2pct 内。

## 部署步骤（PTrade 研究环境）

1. `directives/` 解压到研究环境 `big/directives/`
2. `ptrade_big.py` 上传为策略；`sys_env.txt` 首行 `0`（回测模式，真实下单）
3. 回测建议窗口：**2025-01-02 起**（分钟级校对，对齐主线 wz2 校对窗口）；
   全历史 2013-08-01 起可日频/分钟级长跑。空仓启动即可（首日指导文件全量建仓）
4. 回测后取 `big/big_position_log.json` 三时点快照逐日对账

## 对账口径

- 基准执行价=次日 **09:31 首 bar 价**；PTrade 9:30 窗口下单（9:30~9:31 首根 bar 成交），
  价差为分钟级噪音
- 基准成本：ETF 万5 单边、银行股 千1 单边；PTrade 侧 set_commission 万2.5+最低5元，
  小资金下固定最低佣金会系统性拖累
- 碎单带宽 dust=2%：目标 vs 实际权重差 <2pp 不调仓（两侧同口径）
- 合理偏差源：开盘价差/佣金口径/整手取整；**量级或方向性差异=bug**，
  拿 big_position_log.json 逐日对

## 每日实物流水（holding.py 管线）

`rotation_live.py --run-daily` 每晚：新浪源增量更新 y-data.csv（接缝校验）→
月末自动生成月度 LLM / B 轨持仓周频事件 LLM → 全量重跑 bt_v31 正典 →
`output/live/big_directive_latest.json`（同步 quant_levels/history_output_new/）。
