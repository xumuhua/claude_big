#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动池 月度豆包LLM分析管线 v2 (claude_big) —— 阶段判定+宏观逻辑强度

v2 设计(用户20260806指导): 宏观资产的核心是 阶段(筑底/上升/冲高/下降) × 宏观驱动逻辑
× 逻辑强度。v1 评级式(看多/看空)已证伪——与量化信号信息同源零增量。
v2 的增量在于: ①筑底成功识别(动量未转正前的早鸟信号, 例2026-07黄金)
②事件级强逻辑归因(例2022-02俄乌战争→原油危机) ③逻辑衰竭预警(冲顶离场)。

每月末一次豆包调用: 喂入全池阶段证据特征(截至月末, 无lookahead) + 上月判定(链式),
产出 per-asset {phase, macro_logic, logic_strength, logic_durability} + 池级因子链。
产物: output/llm_monthly/{YYYYMM}.json + {YYYYMM}.md ("version": 2)

用法:
  python3 scripts/monthly_llm_analysis.py --months 202607        # 指定月
  python3 scripts/monthly_llm_analysis.py --all --force          # 全量重跑
  python3 scripts/monthly_llm_analysis.py --all --workers 4
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotation_backtest import TICKERS, CLASSES, load_daily  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "llm_monthly")

MODEL_ID = "bot-20250308205057-5kjf4"
# 与 industry_mainline_scorer_doubao.py:46 同一 fallback (该路径主线复盘60/60零失败)
ARK_API_KEY_FALLBACK = "ec086b7d-fa97-4db2-b196-1eb64c8f5baa"

ASSET_DESC = {
    "513100.SS": "纳指ETF(美股科技成长, QDII, 受美股/汇率/QDII溢价三重驱动)",
    "588000.SS": "科创50ETF(A股硬科技成长, 高beta, 受流动性/产业政策驱动)",
    "518800.SS": "黄金ETF(避险+抗通胀, 受实际利率/美元/地缘风险驱动)",
    # 南方原油501018.SS/嘉实原油160723.SZ: 20260814移出候选池(2027退市)
    # 恒生科技513180.SS: 20260908哥哥指令移出候选池(港股仅留恒生ETF 159920.SZ)
    "601398.SS": "工商银行(类债防御, 高股息, 弱市资金避风港/红利风格)",
    "601988.SS": "中国银行(类债防御, 同上)",
    "601939.SS": "建设银行(类债防御, 同上)",
    "601288.SS": "农业银行(类债防御, 同上)",
    # FIX-HENGKE 20260902: 20260822 三ETF合入TICKERS时漏加描述,
    # build_prompt:ASSET_DESC[t] 每晚KeyError被静默吞 → 事件文档断档15天
    # 588060.SS 科创板ETF: 20260908哥哥指令移出候选池(科创类只留科创50)
    "159920.SZ": "恒生ETF(港股大盘宽基, 受中国政策/美债利率/南向资金驱动)",
    "159952.SZ": "创业板ETF(A股成长宽基, 高beta, 受流动性/成长风格驱动)",
}

# FIX-HENGKE 20260902: 入口断言——池内任何标的缺描述立即失败并点名,
# 而不是在 build_prompt 深处抛裸 KeyError 被调用方静默吞掉
_missing = [t for t in TICKERS if t not in ASSET_DESC]
assert not _missing, f"ASSET_DESC 缺少池内标的描述: {_missing} (新ETF入池必须同步补ASSET_DESC)"


def _ensure_llm_keys():
    """~/keys/*.sh 无 export, source 不进子进程 → 显式解析进 os.environ (仿 replay_llm_pipeline)"""
    if os.environ.get("ARK_API_KEY"):
        return
    for f in ("~/keys/doubao.sh",):
        p = os.path.expanduser(f)
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.replace("export ", "").strip()
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def call_doubao_api(prompt: str, max_retries: int = 3) -> str:
    """原生Ark SDK调用 (仿 industry_mainline_scorer_doubao.py:61-92, 60/60零失败路径)"""
    from volcenginesdkarkruntime import Ark
    client = Ark(api_key=os.environ.get("ARK_API_KEY", ARK_API_KEY_FALLBACK))
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.bot_chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "system", "content": prompt}],
            )
            result = completion.choices[0].message.content.strip()
            if result:
                return result
            raise ValueError("API 返回空文本")
        except Exception as e:
            print(f"    [API重试] 第{attempt}/{max_retries}次失败: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"豆包API调用失败(重试{max_retries}次): {e}")


def _parse_json_response(text: str) -> dict:
    """容错解析 (仿 extract_mainlines._parse_json_response 简化版)"""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    core = text[start:end + 1] if (start != -1 and end > start) else text
    try:
        return json.loads(core)
    except json.JSONDecodeError:
        pass
    fixed = core.replace("“", '"').replace("”", '"').replace("，", ",")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return json.loads(fixed)


# ---------------------------------------------------------------------------
# v2 阶段证据特征构建
# ---------------------------------------------------------------------------

def month_ends(days: pd.DatetimeIndex) -> list:
    """全部月末交易日"""
    s = pd.Series(days, index=days)
    return list(s.groupby([s.index.year, s.index.month]).max())


def build_features(closes: pd.DataFrame, T) -> tuple:
    """截至 T 的全池阶段证据特征表 (DataFrame) + 池级上下文 (dict)

    特征为阶段判定设计(非动量复读):
      位置(回撤/52周分位/低点距今) 趋势结构(vs MA + MA斜率) 节奏(ret加速结构) 波动状态
    """
    c = closes.loc[:T]
    ma20, ma60, ma250 = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(250).mean()
    vol20 = c.pct_change().rolling(20).std() * np.sqrt(244)
    vol120 = c.pct_change().rolling(120).std() * np.sqrt(244)
    rows = []
    for t, name in TICKERS.items():
        px = c[t].dropna()
        if len(px) < 25:                     # vol20需21行; 长窗特征自动为None
            continue
        last = px.iloc[-1]
        def ret(n):
            return round((last / px.iloc[-n] - 1) * 100, 1) if len(px) > n else None
        hi250, lo250 = px.iloc[-250:].max(), px.iloc[-250:].min()
        lo_date = px.iloc[-250:].idxmin()
        lo_days = len(px.loc[lo_date:]) - 1                    # 低点距今交易日数
        pos52 = (last - lo250) / (hi250 - lo250) if hi250 > lo250 else np.nan
        def ma_slope(ma):
            if len(ma[t].dropna()) < 21:
                return None
            return round((ma[t].iloc[-1] / ma[t].iloc[-21] - 1) * 100, 1)
        v20, v120 = vol20[t].iloc[-1], vol120[t].iloc[-1]
        rows.append({
            "代码": t, "名称": name, "类别": CLASSES[t],
            "距250日高点%": round((last / hi250 - 1) * 100, 1),
            "52周区间分位%": round(pos52 * 100, 0) if np.isfinite(pos52) else None,
            "低点距今交易日": lo_days,
            "近5日%": ret(6), "近20日%": ret(21), "近60日%": ret(61), "近120日%": ret(121),
            "vs MA20%": round((last / ma20[t].iloc[-1] - 1) * 100, 1) if np.isfinite(ma20[t].iloc[-1]) else None,
            "vs MA60%": round((last / ma60[t].iloc[-1] - 1) * 100, 1) if np.isfinite(ma60[t].iloc[-1]) else None,
            "vs MA250%": round((last / ma250[t].iloc[-1] - 1) * 100, 1) if np.isfinite(ma250[t].iloc[-1]) else None,
            "MA20斜率%": ma_slope(ma20),
            "MA60斜率%": ma_slope(ma60),
            "vol20%": round(v20 * 100, 1) if np.isfinite(v20) else None,
            "vol120%": round(v120 * 100, 1) if np.isfinite(v120) else None,
            "波动比": round(v20 / v120, 2) if np.isfinite(v20) and np.isfinite(v120) and v120 > 0 else None,
        })
    feat = pd.DataFrame(rows)
    ctx = {
        "n_total": len(feat),
        "above_ma250": int((feat["vs MA250%"] > 0).sum()) if len(feat) else 0,
        "deep_dd": feat.loc[feat["距250日高点%"] <= -20, "名称"].tolist() if len(feat) else [],
    }
    return feat, ctx


def build_prompt(feat: pd.DataFrame, ctx: dict, T, prev: dict | None) -> str:
    asof = T.strftime("%Y-%m-%d")
    table = feat.to_csv(index=False)
    prev_txt = "（本月为首次分析，无上月判定）"
    if prev:
        prev_txt = json.dumps({
            "month": prev.get("month"),
            "phases": {k: v.get("phase") for k, v in prev.get("assets", {}).items()},
            "logics": {k: f"{v.get('macro_logic', '')[:20]}({v.get('logic_strength')})"
                       for k, v in prev.get("assets", {}).items()},
        }, ensure_ascii=False)
    assets_desc = "\n".join(f"- {t} {ASSET_DESC[t]}" for t in TICKERS)
    return f"""你是一名宏观资产配置分析师，管理一个"全天候迷你池"轮动策略，标的如下：
{assets_desc}

【分析基准日】{asof}（月末）。以下阶段证据特征表全部截至该日收盘：
{table}

池级状态：站上MA250的标的 {ctx['above_ma250']}/{ctx['n_total']}；深回撤(≤-20%)标的：{ctx['deep_dd']}。

【上月你的判定】{prev_txt}

【你的任务】宏观资产的涨跌由多月级别的大逻辑驱动（战争/利率周期/汇率/政策/地缘/流动性），
不会像个股一样因单日消息暴动。请对每个标的完成三步判断：

1. **阶段判定** phase ∈ {{筑底, 上升, 冲高, 下降, 震荡}}：
   - 筑底：深回撤后风险释放充分，止跌企稳（低点多日不再创新低、短期收益转正、波动收缩、MA20走平上翘）——即使中长期趋势仍向下
   - 上升：价格站上走升的均线，稳健上涨（涨速可持续，未过度加速）
   - 冲高：远离MA250过度拉伸、短期涨速远超长期、波动放大——泡沫化预警
   - 下降：跌破走降的均线、持续创新低
   - 震荡：无明确方向
2. **宏观逻辑归因** macro_logic：驱动该资产当前趋势的具体宏观事件/因子，**必须点名具体事件**
   （如"俄乌战争推升油价""美联储加息压制成长股""央行购金潮"），不允许写"市场情绪"这类空话。
3. **逻辑强度** logic_strength ∈ {{强, 中, 弱}} 与 **持续性** logic_durability ∈ {{持续, 衰竭}}：
   强=事件级/政策级驱动且仍在发酵（可跟随）；衰竭=逻辑已充分定价或正在逆转（准备离场）。

【纪律要求】
1. 分析基准日为 {asof}，只允许使用以上数据和该日期之前的公开宏观知识；严禁引用之后的实际走势或事件。
2. phase_evidence 必须引用上表数据（如"回撤-25%/低点23日未破/波动比0.6"）。
3. 先自我校验上月判定（阶段判对了吗？逻辑还在吗？误判必须承认），再给本月判断。
4. confidence 取 0~1，表示你对该标的阶段判定的把握。

【新事件扫描（最重要，先做）】在逐项判定前，先回答：过去 1-2 个月内，是否发生了足以改变某资产
定价框架的重大新事件（战争爆发/重大制裁/政策急转/央行非常规行动）？请主动回忆并点名事件与大致时间。
**定价框架转换法则**：若某资产经历长期低位盘整（52周分位<30%、回撤极深）后出现首次强反转，
且恰有重大新事件驱动 → 这是新上升周期的起点，其 logic_strength 应判"强"、logic_durability 判"持续"，
且在该事件未被证伪（停战/政策撤回/官方否认）前，不得因短期盘整或涨幅已高而轻判"衰竭"。
衰竭只用于：事件本身逆转/结束，或价格对利好明显钝化（出利好不再涨）。

【输出格式】严格输出一个 JSON 对象（不要输出其他文字）：
{{
  "new_events": "80字内: 近1-2月改变定价框架的重大新事件及时间(无则写无)",
  "assets": {{
    "513100.SS": {{"phase": "筑底|上升|冲高|下降|震荡",
                  "phase_evidence": "50字内,须引数据",
                  "macro_logic": "50字内,点名具体事件/因子",
                  "logic_strength": "强|中|弱",
                  "logic_durability": "持续|衰竭",
                  "confidence": 0.0}},
    "...所有池中标的代码..."
  }},
  "factor_chains": "100字内: 当前最重要的跨资产因子链条(如 地缘冲突→原油→通胀→利率→成长股承压)",
  "prev_month_review": "80字内",
  "summary": "150字内的本月配置思路"
}}"""


# ---------------------------------------------------------------------------
# 单月处理
# ---------------------------------------------------------------------------

def analyze_month(closes, T, prev, force=False, out_dir=None, tag=None):
    ym = tag or T.strftime("%Y%m")
    od = out_dir or OUT_DIR
    js_path = os.path.join(od, f"{ym}.json")
    md_path = os.path.join(od, f"{ym}.md")
    os.makedirs(od, exist_ok=True)
    if os.path.exists(js_path) and not force:
        old = json.load(open(js_path, encoding="utf-8"))
        if old.get("version") == 2:                # v2 产物才跳过, v1 一律重跑
            return ym, "skip"
    feat, ctx = build_features(closes, T)
    if len(feat) < 3:
        return ym, "too-early"
    prompt = build_prompt(feat, ctx, T, prev)
    raw = call_doubao_api(prompt)
    doc = _parse_json_response(raw)
    doc["version"] = 2
    doc.setdefault("month", ym)
    doc["asof"] = T.strftime("%Y%m%d")
    doc["model"] = MODEL_ID
    # 补齐缺失标的
    for t in feat["代码"]:
        doc.setdefault("assets", {}).setdefault(t, {
            "phase": "震荡", "phase_evidence": "LLM未覆盖", "macro_logic": "",
            "logic_strength": "弱", "logic_durability": "衰竭", "confidence": 0.3})
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    md = [f"# ETF池月度阶段分析 {ym}（基准日 {doc['asof']}，{MODEL_ID}，v2）\n",
          "**因子链**: " + doc.get("factor_chains", "") + "\n",
          "| 标的 | 阶段 | 逻辑(强度/持续) | 证据 | 置信度 |", "|---|---|---|---|---|"]
    for t in feat["代码"]:
        a = doc["assets"][t]
        md.append(f"| {TICKERS[t]}({t}) | {a.get('phase')} | {a.get('macro_logic', '')} "
                  f"({a.get('logic_strength')}/{a.get('logic_durability')}) "
                  f"| {a.get('phase_evidence', '')} | {a.get('confidence')} |")
    md += ["", f"**上月校验**: {doc.get('prev_month_review', '')}", "",
           f"**配置思路**: {doc.get('summary', '')}"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return ym, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", help="YYYYMM 列表")
    ap.add_argument("--dates", nargs="+", help="YYYYMMDD 任意日期列表(事件级分析, 存output/llm_events)")
    ap.add_argument("--all", action="store_true", help="2013-08起全部月末")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", default="2013-08-01")
    args = ap.parse_args()

    _ensure_llm_keys()
    os.makedirs(OUT_DIR, exist_ok=True)
    closes, _ = load_daily()          # 不截断: 首月也需要60日+回溯窗口
    if args.dates:
        # 事件级: 取≤指定日的最近交易日, 以上月月报为链式上文
        ev_dir = os.path.join(ROOT, "output", "llm_events")
        me_all = month_ends(closes.index)
        jobs = []
        for ds in args.dates:
            d = pd.Timestamp(ds)
            T = closes.index[closes.index <= d][-1]
            prev_me = [m for m in me_all if m < T]
            prev = None
            if prev_me:
                p = os.path.join(OUT_DIR, f"{prev_me[-1].strftime('%Y%m')}.json")
                if os.path.exists(p):
                    prev = json.load(open(p))
            jobs.append((T, prev))
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(analyze_month, closes, T, prev, args.force,
                              ev_dir, T.strftime("%Y%m%d")): T for T, prev in jobs}
            for fu in as_completed(futs):
                try:
                    ym, st = fu.result()
                    ok += st in ("ok", "skip")
                    print(f"  {ym}: {st}")
                except Exception as e:
                    fail += 1
                    print(f"  {futs[fu].strftime('%Y%m%d')}: FAIL {e}")
        print(f"事件级完成: ok/skip={ok} fail={fail}")
        return
    me = [T for T in month_ends(closes.index) if T >= pd.Timestamp(args.start)]
    if args.months:
        want = set(args.months)
        me = [T for T in me if T.strftime("%Y%m") in want]
    elif not args.all:
        me = me[-1:]           # 默认只跑最近月末
    # 链式: 按时间序喂上月评级; 并行时各自读盘上已有上月产物
    def prev_of(T):
        idx = month_ends(closes.index)
        i = idx.index(T)
        if i == 0:
            return None
        p = os.path.join(OUT_DIR, f"{idx[i-1].strftime('%Y%m')}.json")
        return json.load(open(p)) if os.path.exists(p) else None

    todo = me
    print(f"待处理 {len(todo)} 个月末 (workers={args.workers})")
    ok = skip = fail = 0
    if args.workers <= 1 or len(todo) <= 1:
        for T in todo:
            try:
                ym, st = analyze_month(closes, T, prev_of(T), args.force)
                print(f"  {ym}: {st}")
                ok += st == "ok"; skip += st != "ok"
            except Exception as e:
                print(f"  {T.strftime('%Y%m')}: FAIL {e}"); fail += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(analyze_month, closes, T, prev_of(T), args.force): T for T in todo}
            for fu in as_completed(futs):
                T = futs[fu]
                try:
                    ym, st = fu.result()
                    print(f"  {ym}: {st}")
                    ok += st == "ok"; skip += st != "ok"
                except Exception as e:
                    print(f"  {T.strftime('%Y%m')}: FAIL {e}"); fail += 1
    print(f"完成: ok={ok} skip={skip} fail={fail} → {OUT_DIR}")


if __name__ == "__main__":
    main()
