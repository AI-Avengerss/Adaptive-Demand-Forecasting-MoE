"""
Step 1: Calendar feature engineering for the M5-based prototype.

This builds the context features the gating network will use later
(volatility proxies will come from sales data once it's uploaded),
plus the calendar-driven signals: event flags, SNAP flags, and a
preliminary day-level "disruption likelihood" tag used to help define
stable vs. volatile windows for the required ablation study.
"""
import pandas as pd
import numpy as np

cal = pd.read_csv("data/calendar.csv")

# --- Basic cleanup ---
cal["date"] = pd.to_datetime(cal["date"])
cal["has_event"] = cal["event_name_1"].notna().astype(int) | cal["event_name_2"].notna().astype(int)
cal["snap_any"] = ((cal["snap_CA"] == 1) | (cal["snap_TX"] == 1) | (cal["snap_WI"] == 1)).astype(int)

# day index as integer for merging with sales columns (d_1, d_2, ...)
cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)

# --- Event-density feature: named events only, in a rolling 7-day window ---
# (SNAP is excluded from this flag: it covers ~10 days/month by design and is too common
# to count as "volatile" on its own. It's kept as a separate feature for the gating network,
# but the regime label should reflect genuinely unusual days, not routine SNAP coverage.)
cal["event_density_7d"] = cal["has_event"].rolling(7, min_periods=1).sum()

# NOTE: calendar_volatile_flag below is a CONTEXT FEATURE for the gating network only.
# It is NOT used as the stable/volatile regime label for the ablation study -- a
# calendar-only label mechanically tags ~45-50% of all days as "volatile" simply because
# a 7-day window around each of 162 event-days covers a large share of the 1,969-day
# calendar. That is a property of the windowing, not a real signal of demand instability,
# and several real products won't actually react to a given generic event at all.
#
# The actual stable/volatile regime split for Objective 3 (the 4-way ablation) will be
# computed from real sales volatility once sales_train_validation.csv is available --
# e.g. periods in the top ~20-25% of rolling demand std flagged as volatile. That is the
# defensible, data-driven label. has_event / snap_any / event_density_7d below remain as
# CONTEXT FEATURES fed into the gating network, which is what the problem statement asks
# for (Objective 2: gating conditioned on "seasonality flags" alongside volatility).
cal["calendar_volatile_flag"] = (cal["event_density_7d"] >= 1).astype(int)

print("Calendar feature summary")
print("-------------------------")
print(f"Total days: {len(cal)}")
print(f"Days with named event: {cal['has_event'].sum()}")
print(f"Days with any SNAP active: {cal['snap_any'].sum()}")
print(f"Days flagged calendar_volatile_flag=1: {cal['calendar_volatile_flag'].sum()} ({cal['calendar_volatile_flag'].mean():.1%})")
print()
print(cal[["date", "d", "d_num", "weekday", "has_event", "snap_any", "calendar_volatile_flag"]].head(15))

cal.to_csv("outputs/calendar_features.csv", index=False)
print("\nSaved -> outputs/calendar_features.csv")
