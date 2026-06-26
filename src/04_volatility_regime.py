"""
Step 4: Compute real, sales-driven volatility and the stable/volatile regime label.

This is the label the 4-way ablation study (Objective 3 of the problem statement)
will actually use -- NOT the calendar-only flag from step 1, which was kept as a
gating-network context feature only (see note in 01_calendar_features.py).

Volatility here is rolling standard deviation of daily sales, per item, computed
over a trailing 14-day window. A day is labeled "volatile" if that item's rolling
std at that point falls in the top quartile of its own historical distribution --
i.e. the threshold is set per-item, not globally, since baseline sales levels
differ a lot across the 20 items (some sell ~5/day, others ~30/day).
"""
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/sales_long_full.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

WINDOW = 14

g = df.groupby("id")["sales"]
df["rolling_std_14"] = g.transform(lambda s: s.rolling(WINDOW, min_periods=3).std())
df["rolling_mean_14"] = g.transform(lambda s: s.rolling(WINDOW, min_periods=3).mean())
df["rolling_cv_14"] = df["rolling_std_14"] / df["rolling_mean_14"].replace(0, np.nan)
df["trend_accel"] = df.groupby("id")["rolling_mean_14"].diff()

# Per-item top-quartile threshold on rolling_cv_14 (coefficient of variation)
thresholds = df.groupby("id")["rolling_cv_14"].quantile(0.75)
df["cv_threshold_75th"] = df["id"].map(thresholds)
df["volatile_regime"] = (df["rolling_cv_14"] > df["cv_threshold_75th"]).astype(int)

# Fill the first WINDOW days (insufficient history) as stable by default -- not enough
# data yet to call them volatile
df["volatile_regime"] = df["volatile_regime"].fillna(0).astype(int)

overall_volatile_share = df["volatile_regime"].mean()
print(f"Overall volatile_regime share: {overall_volatile_share:.1%} (target ~25% by construction)")
print()
print("Per-item volatile share (sanity check -- should hover near 25% each, by design):")
print(df.groupby("id")["volatile_regime"].mean().round(3))

df.to_csv("outputs/sales_with_regime.csv", index=False)
print("\nSaved -> outputs/sales_with_regime.csv")
print("Shape:", df.shape)
