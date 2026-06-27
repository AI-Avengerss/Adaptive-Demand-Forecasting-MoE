"""
Step 2: Reshape M5 wide-format sales data into long format (item, date, sales)
and merge with calendar features built in step 1.
"""
import pandas as pd
import numpy as np

sales = pd.read_csv("data/m5_top80_CA1_FOODS.csv")
cal = pd.read_csv("outputs/calendar_features.csv")
cal["date"] = pd.to_datetime(cal["date"])

day_cols = [c for c in sales.columns if c.startswith("d_")]
id_cols = [c for c in sales.columns if c not in day_cols]

# Melt to long format: one row per (item, day)
long_df = sales.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")

# Merge calendar info (date, event flags, snap flags, etc.) onto each row via the 'd' key
long_df = long_df.merge(
    cal[["d", "date", "wday", "month", "year", "has_event", "snap_CA", "snap_TX", "snap_WI",
         "event_density_7d", "calendar_volatile_flag"]],
    on="d", how="left"
)

# Each store is in CA, so use snap_CA as "snap_active" for this subset
long_df["snap_active"] = long_df["snap_CA"]

long_df["date"] = pd.to_datetime(long_df["date"])
long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)

print("Long-format shape:", long_df.shape)
print("Items:", long_df['id'].nunique(), " | Days per item:", long_df.groupby('id').size().iloc[0])
print(long_df.head(10)[["id", "date", "d", "sales", "wday", "has_event", "snap_active"]])

long_df.to_csv("outputs/sales_long_with_calendar.csv", index=False)
print("\nSaved -> outputs/sales_long_with_calendar.csv")
