#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探查性二级对照：SB后防御期B轨是否让现金留下；不改变正典配置。"""
import itertools
import json
from pathlib import Path

import pandas as pd

import rotation_backtest as rb

ROOT = Path("/data/workspace/quant/sb_hysteresis_20261031")


def main():
    rb.DATA = str(ROOT / "y-data.csv")
    c, o = rb.load_daily()
    c, o = c.loc["2013-08-01":], o.loc["2013-08-01":]
    kw = dict(llm_dir=str(ROOT / "output/llm_monthly"), events_dir=str(ROOT / "output/llm_events"))
    eqs, rows = {}, []
    for bpri, sb_bpri in itertools.product([True, False], repeat=2):
        if sb_bpri is None:
            continue
        label = f"origB{int(bpri)}_defenseB{int(sb_bpri)}"
        eq, tr, w = rb.bt_v31(c, o, sb_days=3, sb_re_days=10, sb_exit_buf=0.97,
                              sb_re_buf=1.0, b_priority=bpri, sb_b_priority=sb_bpri, **kw)
        m = rb.metrics(eq, label)
        eqs[label] = eq
        pd.DataFrame(tr, columns=["date", "type", "ticker", "note"]).to_csv(ROOT / f"trades_{label}.csv", index=False)
        rows.append({"label": label, "original_b_priority": bpri, "sb_b_priority": sb_bpri,
                     "total": m["total"], "mdd": m["mdd"], "sharpe": m["sharpe"], "trades": len(tr)})
        print(label, m, len(tr), flush=True)
    pd.DataFrame(eqs).to_csv(ROOT / "diagnostic_equity.csv")
    (ROOT / "diagnostic_priority.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print("DIAGNOSTIC_DONE")


if __name__ == "__main__":
    main()
