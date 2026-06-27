"""
Step 3: Merge sell_prices into the long-format dataset.

sell_prices is at (store_id, item_id, wm_yr_wk) granularity -- one price per item
per week, not per day. We map each day to its wm_yr_wk via the calendar, then
join the matching weekly price onto every day in that week.
"""
import pandas as pd

long_df = pd.read_csv("outputs/sales_long_with_calendar.csv")
long_df["date"] = pd.to_datetime(long_df["date"])

cal = pd.read_csv("outputs/calendar_features.csv")
cal["date"] = pd.to_datetime(cal["date"])

prices = pd.read_csv("data/sell_prices_top80_CA1.csv")

# Attach wm_yr_wk to each day in long_df
long_df = long_df.merge(cal[["d", "wm_yr_wk"]], on="d", how="left")

# Join weekly price by (store_id, item_id, wm_yr_wk)
long_df = long_df.merge(
    prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

missing = long_df["sell_price"].isna().sum()
print(f"Rows with missing sell_price after merge: {missing} / {len(long_df)} ({missing/len(long_df):.1%})")

# Forward/backward fill within each item in case of any early/late weeks with no listed price
long_df["sell_price"] = (
    long_df.sort_values(["id", "date"])
           .groupby("id")["sell_price"]
           .transform(lambda s: s.ffill().bfill())
)

still_missing = long_df["sell_price"].isna().sum()
print(f"Rows still missing after fill: {still_missing}")

# Price-change feature: week-over-week pct change, a genuinely useful gating context signal
long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)
long_df["price_pct_change"] = long_df.groupby("id")["sell_price"].pct_change().fillna(0)

print(long_df.head(8)[["id", "date", "sales", "sell_price", "price_pct_change"]])

long_df.to_csv("outputs/sales_long_full.csv", index=False)
print("\nSaved -> outputs/sales_long_full.csv")
print("Final shape:", long_df.shape)
