# -*- coding: utf-8 -*-
# ============================================================================
# claude_big ETF轮动 v3.10 · PTrade 实盘/回测策略（决策-执行分离 + 目标仓位驱动）
#
# 架构（与主线 ptrade_wz2 同框架，算法与交易隔离）：
#   Linux守备进程(holding.py → rotation_live.py) 每晚全量重跑 bt_v31 正典配置,
#   产出 big_directive 交易指导文件（=次日开盘的完整目标权重快照, 10票池:
#   纳指/恒生科技/科创50/黄金/原油×2/四大行）
#     │
#     ├─ market_en=0 回测模式：读按日归档指导文件，真实下单（平台内验证执行损耗）
#     └─ market_en=1 部署模式：读 latest 指导文件，【只输出目标仓位+动作建议】
#         到 hit_stocks/（与ptrade_demon/ptrade_wz2同格式），交易守备进程真实下单
#
# 执行模型（与 wz2 的差别）：
#   - 指导文件是【每日全量目标权重快照】（wz2是增量行业篮子），盘前整体替换目标表
#   - 回测口径=信号日15:00判定, 次日09:31开盘成交（买卖同窗）→ 9:30窗口先卖后买
#   - 差额调仓：目标vs实际权重差≥2%(dust, 与回测引擎同口径)才动;
#     持仓不在目标 → 全清; 在目标欠配 → 补差额; 超配 → 减到目标
#   - 顺延自洽：涨停撤全新建仓/跌停停牌顺延, 次日指导文件若口径不变自动重试
#     （目标表由每日指导文件重建, 无需持久化持仓状态, 状态丢失天然自愈）
#
# 时序纪律：
#   9:30 买入+卖出窗口（930~931首根bar触发）：先卖(释放现金)后买;
#        涨停撤单(开盘涨停=放弃该票本次建仓,仅全新建仓生效)、跌停/停牌卖出顺延、
#        整手100份、5万股拆单、现金约束内按权重降序；未成交委托不撤销留当日撮合
#   10:00 确认快照；收盘快照+状态存盘
#
# 重启容错（平台只会在 after_trading_end 之后重启）：
#   big_state.json 仅存名称表/最近指导日期(目标表每日由指导文件全量重建,
#   文件缺失则沿用昨日目标, 绝不因状态丢失误清仓)。
# ============================================================================
import json
import datetime
import numpy as np
import pandas as pd


def initialize(context):
    g.notebook_path = get_research_path() + "/big/"
    g.upload_path = get_research_path() + "/upload_file/"
    g.hit_stocks_path = get_research_path() + "/hit_stocks/"
    set_commission(commission_ratio=0.00025, min_commission=5, type="STOCK")
    set_volume_ratio(volume_ratio=1)
    g.strategy_name = "big_rotation"
    log.info("begin " + g.strategy_name)

    # 环境开关：0=回测模式(真实下单) 1=部署模式(仅输出建议, 交易守备进程执行)
    inf = open(get_research_path() + '/sys_env.txt', 'r')
    line = inf.readline()
    if line[-1] == "\n":
        line = line[:-1]
    g.market_en = int(line.split(",")[0])
    log.info("market en\t" + str(g.market_en))

    g.trade_time = 930    # 调仓窗口（买卖同窗, 对齐回测"次日09:31开盘成交"口径）
    g.dust = 0.02         # 碎单过滤带宽（与 bt_v31 dust=0.02 同口径）
    g.target = {}         # 目标持仓 code6 -> {"weight","name"}
    g.names = {}          # code6 -> 名称
    g.actions = []        # 当日动作清单（部署模式advice用）
    g.directive_date = "" # 最近应用的指导文件信号日


# ---------- 工具 ----------

def cvt_to_ptrade_code(code):
    """指导文件代码已是 ptrade 后缀格式 (513100.SS/518800.SS), 容错6位裸码。
    (旧示例160723.SZ为嘉实原油, 20260814随2027退市移出候选池, 此处仅作格式示例)"""
    if "." in code:
        return code
    if code[:2] in ("50", "51", "58", "60"):
        return code[:6] + ".SS"
    return code[:6] + ".SZ"


def get_limit_pct(code, name):
    """板块涨跌停幅度（小数）。本池：588科创ETF=20%, 其余ETF/LOF/银行股=10%。"""
    if str(code)[:3] == "588":
        return 0.198
    return 0.098


def _load_directive():
    """读取指导文件：部署模式=latest；回测模式=按前一交易日日期归档。"""
    if g.market_en == 1:
        path = g.upload_path + "big_directive_latest.json"
    else:
        day_time = str(get_trading_day(day=-1))
        day_time = day_time[:4] + day_time[5:7] + day_time[8:]
        path = g.notebook_path + "directives/big_directive_" + day_time + ".json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        log.info("directive loaded: " + path
                 + " signal=" + str(doc.get("signal_date"))
                 + " exec=" + str(doc.get("exec_date")))
        return doc
    except Exception as e:
        log.info("directive load FAIL: " + path + " " + str(e))
        return None


def _save_state():
    doc = {"names": g.names, "directive_date": g.directive_date,
           "target": g.target}
    with open(g.notebook_path + "big_state.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


def _load_state():
    try:
        with open(g.notebook_path + "big_state.json", "r", encoding="utf-8") as f:
            doc = json.load(f)
        g.names = doc.get("names", {})
        g.directive_date = doc.get("directive_date", "")
        if not g.target:            # 指导文件缺失时的昨日目标兜底
            g.target = doc.get("target", {})
        log.info("state loaded: last_directive=" + str(g.directive_date))
        return True
    except Exception as e:
        log.info("state load FAIL (init empty): " + str(e))
        return False


def _record_action(time_tag, side, code, weight, reason):
    g.actions.append({"time": time_tag, "side": side, "code": code,
                      "name": g.names.get(code, ""),
                      "weight": weight, "reason": reason})
    log.info(",".join(["advice" if g.market_en == 1 else "trade",
                       time_tag, side, code, g.names.get(code, ""),
                       str(round(weight, 4)), reason]))


def _pc(data, sec, code):
    """交易所昨收：优先行情对象自带preclose；后备盘前get_history原始收盘。"""
    try:
        v = data[sec].preclose
        if v and v > 0:
            return float(v)
    except Exception:
        pass
    return g.preclose.get(code, 0)


def _dump_positions(context, tag, day_time, format_time):
    """持仓快照写文件（debug用）：open=9:30调仓前, open30=10:00确认, close=收盘。
    JSON Lines追加到 big/big_position_log.json，回测后逐日对账。"""
    pv = context.portfolio.portfolio_value
    cash = context.portfolio.cash
    positions = context.portfolio.positions
    rows = []
    if g.market_en == 1:
        for code, t in g.target.items():
            if t.get("weight", 0) <= 0:
                continue
            rows.append({
                "code": code, "name": g.names.get(code, ""),
                "amount": None, "enable_amount": None,
                "price": None, "market_value": None,
                "weight": round(t.get("weight", 0), 4),
                "cost_basis": None, "pnl_pct": None,
                "virtual": True,
            })
    for sec in positions:
        pos = positions[sec]
        code = sec[:6]
        price = 0.0
        try:
            price = float(pos.last_sale_price)
        except Exception:
            pass
        mv = pos.amount * price
        cost = 0.0
        for attr in ("cost_basis", "avg_cost", "hold_cost"):
            try:
                v = getattr(pos, attr)
                if v and v > 0:
                    cost = float(v)
                    break
            except Exception:
                pass
        pnl_pct = (price / cost - 1) if cost > 0 and price > 0 else None
        rows.append({
            "code": code, "name": g.names.get(code, ""),
            "amount": pos.amount, "enable_amount": pos.enable_amount,
            "price": round(price, 3),
            "market_value": round(mv, 2),
            "weight": round(mv / pv, 4) if pv > 0 else 0,
            "cost_basis": round(cost, 3),
            "pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
        })
    rec = {"date": day_time, "tag": tag, "time": format_time,
           "portfolio_value": round(pv, 2), "cash": round(cash, 2),
           "invested_weight": round(sum(r["weight"] for r in rows), 4),
           "n_positions": len(rows), "target_count": len(g.target),
           "positions": rows}
    try:
        with open(g.notebook_path + "big_position_log.json", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.info("dump_positions fail: " + str(e))
    log.info("pos_dump," + tag + "," + str(day_time)
             + ",n=" + str(len(rows))
             + ",invested=" + str(rec["invested_weight"]))


def _write_advice(context, day_time):
    """部署模式：输出目标持仓+动作清单（交易守备进程消费）。
    目标持仓文件格式与ptrade_demon/ptrade_wz2完全一致(stock/weight/force_trade
    + param.txt)，保证平台既有守备进程可直接接管。"""
    hit_stocks = []
    hit_weights = []
    force_trade = []
    for code, pos in g.target.items():
        if pos.get("weight", 0) > 0:
            hit_stocks.append(cvt_to_ptrade_code(code))
            hit_weights.append(pos["weight"])
            force_trade.append(1 if g.directive_date == day_time else 0)
    # 不写weight=0平仓行：守备进程发现实盘有持仓但不在建议清单时，
    # 会自动按weight=0计算差额并卖出至空仓（用户20260730确认）
    # 与demon口径一致：除以max(1.0, sum)——不满仓时保留真实仓位比例
    sum_weights = sum(hit_weights)
    hit_weights = [w / max(1.0, sum_weights) for w in hit_weights]
    hit_df = pd.DataFrame({"stock": hit_stocks, "weight": hit_weights,
                           "force_trade": force_trade})
    if len(hit_stocks):
        hit_df.set_index("stock", inplace=True)
    hit_df.to_csv(g.hit_stocks_path + g.strategy_name + ".txt")
    param_f = open(g.hit_stocks_path + g.strategy_name + ".param.txt", "w+")
    param_f.write("1.0\n")
    param_f.close()
    advice = {"date": day_time, "strategy": g.strategy_name,
              "mode": "big",
              "target": [{"code": c, "weight": round(w, 4)}
                         for c, w in zip(hit_stocks, hit_weights)],
              "actions": g.actions}
    with open(g.hit_stocks_path + g.strategy_name + "_actions_" + day_time + ".json",
              "w", encoding="utf-8") as f:
        json.dump(advice, f, ensure_ascii=False, indent=1)


# ---------- 盘前：应用指导文件到目标持仓 ----------

def before_trading_start(context, data):
    now = context.blotter.current_dt
    day_time = now.strftime('%Y%m%d')
    _load_state()
    g.prev_target = dict(g.target)      # 昨日目标(部署模式卖出建议对照用)
    g.directive = _load_directive()
    g.actions = []
    g.preclose = {}

    # ===== 应用指导文件：全量目标权重快照整体替换 =====
    if g.directive is not None:
        if str(g.directive.get("exec_date")) != day_time:
            log.info("WARNING: directive exec_date="
                     + str(g.directive.get("exec_date")) + " today=" + day_time)
        new_target = {}
        for t in g.directive.get("targets", []):
            code = t["code"][:6]
            new_target[code] = {"weight": t.get("weight", 0),
                                "name": t.get("name", "")}
            g.names[code] = t.get("name", g.names.get(code, ""))
        g.target = new_target
        g.directive_date = str(g.directive.get("signal_date", ""))
    else:
        # 指导文件缺失：沿用状态里昨日目标（不调仓），绝不误判清仓
        log.info("directive missing, keep yesterday target: "
                 + str(len(g.target)))

    # 数据宇宙：目标 + 实盘持仓
    universe = set(g.target.keys())
    if g.market_en == 0:
        universe.update(s[:6] for s in context.portfolio.positions)
    universe = sorted(universe)
    g.universe = universe

    if universe:
        sec_list = [cvt_to_ptrade_code(c) for c in universe]
        try:
            hdf = get_history(security_list=sec_list, count=1,
                              frequency='1d', fq=None)
            hdf = hdf.set_index('code')
            for c in universe:
                sec = cvt_to_ptrade_code(c)
                try:
                    g.preclose[c] = float(hdf.loc[sec, 'close'])
                except Exception:
                    g.preclose[c] = 0.0
        except Exception as e:
            log.info("preclose load fail: " + str(e))

    log.info("before_trading_start done: target=" + str(len(g.target))
             + " universe=" + str(len(universe)))


# ---------- 9:30 调仓窗口：先卖后买，差额调仓 ----------

def do_rebalance(context, data, day_time):
    positions = context.portfolio.positions
    pv = context.portfolio.portfolio_value

    # ----- 1) 卖出：不在目标的全清 + 超配减仓（释放现金） -----
    if g.market_en == 0:
        for sec in list(positions.keys()):
            code = sec[:6]
            pos = positions[sec]
            amt = pos.enable_amount
            if amt <= 0:
                continue                      # T+1未解禁
            try:
                px = data[sec].open
                if px <= 0:
                    continue
            except Exception:
                continue
            cur_w = pos.amount * px / pv if pv > 0 else 0
            tgt_w = g.target.get(code, {}).get("weight", 0)
            if code in g.target and tgt_w > 0:
                if cur_w - tgt_w < g.dust:
                    continue                  # 超配在带宽内：不动
                sell_amt = int((cur_w - tgt_w) * pv / px / 100) * 100
                sell_amt = min(sell_amt, amt)
                reason = "超配减仓至目标"
            else:
                sell_amt = amt                # 不在目标：全清
                reason = "跌出目标持仓"
            if sell_amt < 100:
                continue
            try:
                if data[sec].is_open == 0:
                    log.info("sell_skip_suspend," + str(day_time) + "," + code)
                    continue
            except Exception:
                continue
            pc = _pc(data, sec, code)
            lim = get_limit_pct(code, g.names.get(code, ""))
            if pc > 0 and data[sec].open <= pc * (1 - lim + 0.001):
                log.info("sell_defer_limitdown," + str(day_time) + "," + code)
                continue                      # 跌停：顺延, 次日指导文件不变则重试
            remain = sell_amt
            while remain > 0:                 # 5万股拆单（demon执行框架件）
                h = min(50000, remain)
                order(sec, -h)
                remain -= h
            _record_action("0930", "sell", code, tgt_w, reason)
    else:
        # 部署模式：不交易，登记退出建议（守备进程按hit清单自动算weight=0差额，
        # 此处仅为 actions 留痕）
        for code in getattr(g, "prev_target", {}):
            if code not in g.target:
                _record_action("0930", "sell", code, 0, "跌出目标持仓")

    # ----- 2) 买入：欠配补差额（按权重降序，现金约束） -----
    cash_avail = context.portfolio.cash
    for code, t in sorted(g.target.items(),
                          key=lambda kv: -kv[1].get("weight", 0)):
        weight = t.get("weight", 0)
        if weight <= 0:
            continue
        sec = cvt_to_ptrade_code(code)
        cur_mv = 0.0
        if g.market_en == 0 and sec in positions and positions[sec].amount > 0:
            try:
                cur_mv = positions[sec].amount * data[sec].open
            except Exception:
                cur_mv = pv * weight        # 取价失败视为已达目标, 不折腾
        tgt_mv = pv * weight
        if tgt_mv - cur_mv < g.dust * pv:
            continue                          # 欠配在带宽内：不动
        pc = _pc(data, sec, code)
        if pc <= 0:
            log.info("buy_skip_no_preclose," + str(day_time) + "," + code)
            continue
        try:
            if data[sec].is_open == 0:
                log.info("buy_skip_suspend," + str(day_time) + "," + code)
                continue
        except Exception:
            log.info("buy_skip_no_data," + str(day_time) + "," + code)
            continue
        op = data[sec].open
        if op <= 0:
            log.info("buy_skip_zero_open," + str(day_time) + "," + code)
            continue
        lim = get_limit_pct(code, g.names.get(code, ""))
        if cur_mv <= 0 and op >= pc * (1 + lim - 0.001):
            # 涨停撤单：不追高, 放弃本次建仓（仅全新建仓生效）;
            # 目标表仍在指导文件里, 次日指导口径不变则自动重试
            log.info("buy_cancel_limitup," + str(day_time) + "," + code
                     + ",open=" + str(op) + ",pc=" + str(pc))
            continue
        if g.market_en == 0:
            buy_value = tgt_mv - cur_mv
            hands = int(buy_value / op / 100) * 100
            if hands * op > cash_avail:
                hands = int(cash_avail / op / 100) * 100
            if hands <= 0:
                log.info("buy_skip_cash," + str(day_time) + "," + code
                         + ",cash=" + str(int(cash_avail)))
                continue
            remain = hands
            while remain > 0:
                h = min(50000, remain)
                order(sec, h)
                remain -= h
            cash_avail -= hands * op
        if cur_mv <= 0:
            _record_action("0930", "buy", code, weight, "建仓/补仓至目标")
        else:
            _record_action("0930", "buy_topup", code, weight, "欠配补差额")


# ---------- 主循环 ----------

def handle_data(context, data):
    now = context.blotter.current_dt
    format_time = int(now.strftime('%H%M'))
    day_time = now.strftime('%Y%m%d')

    # 调仓窗口 9:30~9:31（部分平台分钟线首根bar是9:31，窗口+一次性标志双保险）
    if g.trade_time <= format_time <= 931 and not getattr(g, "_rb_done_" + day_time, False):
        setattr(g, "_rb_done_" + day_time, True)
        _dump_positions(context, "open", day_time, format_time)   # 调仓前快照
        do_rebalance(context, data, day_time)
        if g.market_en == 1:
            _write_advice(context, day_time)

    # 开盘半小时 10:00~10:01：确认开盘调仓成交（debug快照）
    if 1000 <= format_time <= 1001 and not getattr(g, "_chk_done_" + day_time, False):
        setattr(g, "_chk_done_" + day_time, True)
        _dump_positions(context, "open30", day_time, format_time)


def after_trading_end(context, data):
    now = context.blotter.current_dt
    day_time = now.strftime('%Y%m%d')
    _dump_positions(context, "close", day_time, 1500)   # 收盘快照
    _save_state()
    if g.market_en == 1:
        _write_advice(context, day_time)
    line = "结束总值: " + str(context.portfolio.portfolio_value) \
        + " 现金 " + str(context.portfolio.cash) \
        + " 目标持仓 " + str(len(g.target))
    log.info(line)
