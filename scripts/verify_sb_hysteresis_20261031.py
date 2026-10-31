#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归验证：新研究参数不改变默认正典；旧版SB(默认水线)行为不变。"""
import os
from pathlib import Path

import pandas as pd

import rotation_backtest as rb

ROOT = Path(os.environ["SB_FROZEN_ROOT"])
rb.DATA = str(ROOT / "y-data.csv")
c, o = rb.load_daily()
c, o = c.loc["2013-08-01":], o.loc["2013-08-01":]
kw = dict(llm_dir=str(ROOT / "output/llm_monthly"), events_dir=str(ROOT / "output/llm_events"))
eq, tr, w = rb.bt_v31(c, o, **kw)
base = pd.read_csv(ROOT / "baseline_equity.csv", index_col=0, parse_dates=True).iloc[:, 0]
base_tr = pd.read_csv(ROOT / "baseline_trades.csv")
pd.testing.assert_series_equal(eq, base, check_names=False)
assert [tuple(x) for x in tr] == list(base_tr.itertuples(index=False, name=None))
eq2, tr2, w2 = rb.run_v31_canonical(c, o)
pd.testing.assert_series_equal(eq, eq2)
pd.testing.assert_frame_equal(w, w2)
assert tr == tr2
# 原始实现来源: git show HEAD:scripts/rotation_backtest.py
legacy_path = ROOT / "rotation_backtest_legacy.py"
import importlib.util
spec = importlib.util.spec_from_file_location("legacy", legacy_path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
legacy.DATA = str(ROOT / "y-data.csv")
lc, lo = legacy.load_daily()
lc, lo = lc.loc["2013-08-01":], lo.loc["2013-08-01":]
leq, ltr, lw = legacy.bt_v31(lc, lo, sb_days=5, sb_re_days=5, **kw)
neq, ntr, nw = rb.bt_v31(c, o, sb_days=5, sb_re_days=5, **kw)
pd.testing.assert_series_equal(leq, neq)
pd.testing.assert_frame_equal(lw, nw)
assert ltr == ntr
print("LEGACY_SB_5_5_REGRESSION_PASS", float(neq.iloc[-1]), len(ntr))
# 双水线边界：低位MA250=1，出场线0.97，回归线1.01。
small_days = pd.bdate_range("2024-01-01", periods=200)
sc = pd.DataFrame(1.0, index=small_days, columns=list(rb.TICKERS))
so = sc.copy()
ndx_col = list(rb.TICKERS).index("513100.SS")
sc.iloc[:3, ndx_col] = 0.96
sc.iloc[3, ndx_col] = 1.01
sc.iloc[4, ndx_col] = 1.005
orig_roll = pd.DataFrame.rolling
orig_mean = pd.core.window.rolling.Rolling.mean
fixed_ma = pd.DataFrame(1.0, index=small_days, columns=list(rb.TICKERS))
fixed_hi = fixed_ma.copy()

def five_day_ma(self, *a, **kw):
    r = orig_roll(self, *a, **kw)
    if a and a[0] == 250:
        return type("MA", (), {
            "mean": lambda s: fixed_ma,
            "max": lambda s: fixed_hi,
        })()
    return r

pd.DataFrame.rolling = five_day_ma
try:
    _, strades, _ = rb.bt_v31(sc, so, sb_days=3, sb_re_days=2,
                              sb_exit_buf=0.97, sb_re_buf=1.01,
                              use_events=False, use_llm_exit=False, grid_on=False,
                              nuke_stop=0.0)
finally:
    pd.DataFrame.rolling = orig_roll
sbtr = [t for t in strades if t[1] == "SB-OUT"]
assert len(sbtr) == 1, strades
print("HYSTERESIS_MID_BAND_RESET_PASS", sbtr)
# 宽回归水线+高出场水线: 换防后若还在出场线下，仍应能按回归线恢复。
sc2 = sc.copy()
sc2.iloc[3, ndx_col] = 1.02
sc2.iloc[4, ndx_col] = 1.02
pd.DataFrame.rolling = five_day_ma
try:
    _, strades2, _ = rb.bt_v31(sc2, so, sb_days=3, sb_re_days=2,
                               sb_exit_buf=0.98, sb_re_buf=1.01,
                               use_events=False, use_llm_exit=False, grid_on=False,
                               nuke_stop=0.0)
finally:
    pd.DataFrame.rolling = orig_roll
assert [t[1] for t in strades2 if t[1].startswith("SB-")] == ["SB-OUT", "SB-IN"], strades2
print("WIDE_REENTRY_ABOVE_EXIT_LINE_PASS", [t for t in strades2 if t[1].startswith("SB-")])
print("DEFAULT_CANONICAL_REGRESSION_PASS", float(eq.iloc[-1]), len(tr))
print("LEGACY_SB_5_5_REGRESSION_PASS", float(neq.iloc[-1]), len(ntr))
