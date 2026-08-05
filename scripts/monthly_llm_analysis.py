#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动池 月度豆包LLM分析管线 (claude_big)

每月末一次豆包调用: 喂入全池量化特征(截至月末, 无lookahead) + 上月评级(链式连续),
产出 资产级评级(看多/中性/看空+置信度) + 整体风险偏好(risk_on/neutral/risk_off),
供 rotation_backtest.py --llm-mode 挂载指导交易。

产物: output/llm_monthly/{YYYYMM}.json + {YYYYMM}.md

用法:
  python3 scripts/monthly_llm_analysis.py --months 202607        # 指定月
  python3 scripts/monthly_llm_analysis.py --all                  # 2013-08起全部月末(增量)
  python3 scripts/monthly_llm_analysis.py --all --force          # 强制重跑
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
    "513180.SS": "恒生科技ETF(港股科技, 受中国政策/美债利率/南向资金驱动)",
    "588000.SS": "科创50ETF(A股硬科技成长, 高beta, 受流动性/产业政策驱动)",
    "518800.SS": "黄金ETF(避险+抗通胀, 受实际利率/美元/地缘风险驱动)",
    "501018.SS": "南方原油(商品QDII, 受OPEC/库存/地缘驱动, 注意移仓损耗)",
    "160723.SZ": "嘉实原油(商品QDII, 同上)",
    "601398.SS": "工商银行(类债防御, 高股息, 弱市资金避风港/红利风格)",
    "601988.SS": "中国银行(类债防御, 同上)",
    "601939.SS": "建设银行(类债防御, 同上)",
    "601288.SS": "农业银行(类债防御, 同上)",
}


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
# 特征构建
# ---------------------------------------------------------------------------

def month_ends(days: pd.DatetimeIndex) -> list:
    """全部月末交易日"""
    s = pd.Series(days, index=days)
    return list(s.groupby([s.index.year, s.index.month]).max())


def build_features(closes: pd.DataFrame, T) -> tuple:
    """截至 T 的全池特征表 (DataFrame) + 池级上下文 (dict)"""
    c = closes.loc[:T]
    vol20 = c.pct_change().rolling(20).std() * np.sqrt(244)
    score = (0.5 * (c / c.shift(20) - 1) + 0.5 * (c / c.shift(60) - 1)) / vol20.replace(0, np.nan)
    ma60, ma250 = c.rolling(60).mean(), c.rolling(250).mean()
    rows = []
    for t, name in TICKERS.items():
        px = c[t].dropna()
        if len(px) < 30:
            continue
        last = px.iloc[-1]
        def ret(n):
            return (last / px.iloc[-n] - 1) if len(px) >= n else np.nan
        hi60 = px.iloc[-60:].max()
        rows.append({
            "代码": t, "名称": name, "类别": CLASSES[t],
            "1月收益%": round(ret(22) * 100, 1),
            "3月收益%": round(ret(66) * 100, 1) if len(px) >= 66 else None,
            "6月收益%": round(ret(132) * 100, 1) if len(px) >= 132 else None,
            "12月收益%": round(ret(244) * 100, 1) if len(px) >= 244 else None,
            "年化波动%": round(vol20[t].iloc[-1] * 100, 1),
            "距60日高点%": round((last / hi60 - 1) * 100, 1),
            "vs MA60%": round((last / ma60[t].iloc[-1] - 1) * 100, 1) if np.isfinite(ma60[t].iloc[-1]) else None,
            "vs MA250%": round((last / ma250[t].iloc[-1] - 1) * 100, 1) if np.isfinite(ma250[t].iloc[-1]) else None,
            "动量分": round(score[t].iloc[-1], 2),
        })
    feat = pd.DataFrame(rows)
    feat["动量分"] = pd.to_numeric(feat["动量分"], errors="coerce")
    feat["动量排名"] = feat["动量分"].rank(ascending=False, method="first").astype("Int64")
    ctx = {
        "n_pass": int((feat["动量分"] > 0).sum()),
        "n_total": len(feat),
        "top": feat.sort_values("动量分", ascending=False).head(3)[["名称", "动量分"]].values.tolist(),
    }
    return feat, ctx


def build_prompt(feat: pd.DataFrame, ctx: dict, T, prev: dict | None) -> str:
    asof = T.strftime("%Y-%m-%d")
    table = feat.to_csv(index=False)
    prev_txt = "（本月为首次分析，无上月评级）"
    if prev:
        prev_txt = json.dumps({
            "month": prev.get("month"), "regime": prev.get("regime"),
            "ratings": {k: v.get("rating") for k, v in prev.get("assets", {}).items()},
        }, ensure_ascii=False)
    assets_desc = "\n".join(f"- {t} {ASSET_DESC[t]}" for t in TICKERS)
    return f"""你是一名宏观资产配置分析师，管理一个"全天候迷你池"轮动策略，标的如下：
{assets_desc}

【分析基准日】{asof}（月末）。以下量化特征表全部截至该日收盘：
{table}

池级状态：动量分>0 的标的 {ctx['n_pass']}/{ctx['n_total']}；动量前三：{ctx['top']}。

【上月你的评级】{prev_txt}

【纪律要求】
1. 分析基准日为 {asof}，只允许使用以上数据和该日期之前的公开宏观知识；严禁引用该日期之后的实际走势或事件。
2. 每个评级必须引用上表数据佐证（如"3月收益-12%且跌破MA250"），不允许空泛判断。
3. 先自我校验上月评级（对了什么、错了什么、教训），再给本月判断。
4. 评级含义："看多"=未来1-3个月预期正收益且风险可控；"看空"=预期下跌或风险远大于机会；"中性"=方向不明或赔率一般。confidence 取 0~1。
5. regime 定义：risk_on=风险资产整体顺风可积极；neutral=结构分化正常轮动；risk_off=宏观环境恶劣（流动性冲击/系统性风险/多数资产破位），应大幅降低权益仓位。

【输出格式】严格输出一个 JSON 对象（不要输出其他文字）：
{{
  "regime": "risk_on|neutral|risk_off",
  "regime_reason": "80字内",
  "assets": {{
    "513100.SS": {{"rating": "看多|中性|看空", "confidence": 0.0, "reason": "60字内,须引数据"}},
    "...所有池中标的代码..."
  }},
  "prev_month_review": "80字内",
  "summary": "150字内的本月配置思路"
}}"""


# ---------------------------------------------------------------------------
# 单月处理
# ---------------------------------------------------------------------------

def analyze_month(closes, T, prev, force=False):
    ym = T.strftime("%Y%m")
    js_path = os.path.join(OUT_DIR, f"{ym}.json")
    md_path = os.path.join(OUT_DIR, f"{ym}.md")
    if os.path.exists(js_path) and not force:
        return ym, "skip"
    feat, ctx = build_features(closes, T)
    if len(feat) < 3:
        return ym, "too-early"
    prompt = build_prompt(feat, ctx, T, prev)
    raw = call_doubao_api(prompt)
    doc = _parse_json_response(raw)
    doc.setdefault("month", ym)
    doc["asof"] = T.strftime("%Y%m%d")
    doc["model"] = MODEL_ID
    # 补齐缺失标的为中性
    for t in feat["代码"]:
        doc.setdefault("assets", {}).setdefault(t, {"rating": "中性", "confidence": 0.5, "reason": "LLM未覆盖"})
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    md = [f"# ETF池月度分析 {ym}（基准日 {doc['asof']}，{MODEL_ID}）\n",
          f"**regime**: {doc.get('regime')} — {doc.get('regime_reason', '')}\n",
          "| 标的 | 评级 | 置信度 | 理由 |", "|---|---|---|---|"]
    for t in feat["代码"]:
        a = doc["assets"][t]
        md.append(f"| {TICKERS[t]}({t}) | {a.get('rating')} | {a.get('confidence')} | {a.get('reason', '')} |")
    md += ["", f"**上月校验**: {doc.get('prev_month_review', '')}", "",
           f"**配置思路**: {doc.get('summary', '')}"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return ym, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", help="YYYYMM 列表")
    ap.add_argument("--all", action="store_true", help="2013-08起全部月末")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", default="2013-08-01")
    args = ap.parse_args()

    _ensure_llm_keys()
    os.makedirs(OUT_DIR, exist_ok=True)
    closes, _ = load_daily()          # 不截断: 首月也需要60日+回溯窗口
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
