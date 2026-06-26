"""
Step 5: Build the supervised learning dataset.

Target: predict sales on day t, using only information available up to day t-1
(strict no-leakage lag construction).

For LightGBM: tabular lag + rolling features.
For LSTM: a separate sequence-windowed array (built in step 6).

Train/val/test split: chronological (not random) since this is a time series --
last 28 days held out as test (matches M5's own validation convention), the 28
days before that as validation, everything else as train.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/sales_with_regime.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# --- Lag features (all strictly using data up to t-1, no leakage) ---
g = df.groupby("id")["sales"]
for lag in [1, 2, 3, 7, 14]:
    df[f"sales_lag_{lag}"] = g.shift(lag)

# Rolling stats computed on lag_1 onward (i.e. excludes day t itself)
df["roll_mean_7_lag1"]  = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(7,  min_periods=3).mean())
df["roll_std_7_lag1"]   = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(7,  min_periods=3).std())
df["roll_mean_14_lag1"] = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
df["roll_std_14_lag1"]  = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(14, min_periods=3).std())

# Calendar/price features are already known in advance (no leakage risk): wday, month,
# has_event, snap_active, sell_price, price_pct_change -- these stay as-is.

# Drop the first 14 rows per item (insufficient lag history)
df["_rownum"] = df.groupby("id").cumcount()
df = df[df["_rownum"] >= 14].drop(columns="_rownum").reset_index(drop=True)

print("After dropping warm-up rows:", df.shape)
print("Remaining NaNs in feature columns:")
feat_check = ["sales_lag_1","sales_lag_7","roll_mean_7_lag1","roll_std_7_lag1","roll_mean_14_lag1","roll_std_14_lag1"]
print(df[feat_check].isna().sum())

# Any remaining NaNs (e.g. std on a constant short window) -> fill with 0
df[feat_check] = df[feat_check].fillna(0)

# --- Chronological split ---
max_date = df["date"].max()
test_start = max_date - pd.Timedelta(days=27)
val_start = test_start - pd.Timedelta(days=28)

df["split"] = np.where(df["date"] >= test_start, "test",
                 np.where(df["date"] >= val_start, "val", "train"))

print()
print(df["split"].value_counts())
print()
print("Date ranges per split:")
print(df.groupby("split")["date"].agg(["min", "max"]))

df.to_csv("outputs/model_dataset.csv", index=False)
print("\nSaved -> outputs/model_dataset.csv")
