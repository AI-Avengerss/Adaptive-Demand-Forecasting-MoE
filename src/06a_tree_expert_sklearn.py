"""
Step 6a: Train the tree-based expert.

SANDBOX NOTE: LightGBM is not installable in this environment (no network access
for pip). This uses sklearn's HistGradientBoostingRegressor as a direct stand-in --
same family of model (histogram-based gradient boosted trees), same strengths
(nonlinear feature interactions, fast on tabular lag features, no long-sequence
memory). The real LightGBM training script (functionally equivalent, same feature
set) is provided separately as 06b_lightgbm_real.py to run on a machine with
internet access; it reads the same model_dataset.csv and produces the same
prediction columns, so it's a drop-in replacement.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("outputs/model_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

FEATURES = [
    "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_7", "sales_lag_14",
    "roll_mean_7_lag1", "roll_std_7_lag1", "roll_mean_14_lag1", "roll_std_14_lag1",
    "wday", "month", "has_event", "snap_active", "sell_price", "price_pct_change",
]
TARGET = "sales"

# One-hot the item id so a single shared model can specialize per item via this feature
df = pd.get_dummies(df, columns=["id"], prefix="idflag")
item_cols = [c for c in df.columns if c.startswith("idflag_")]
FEATURES_FULL = FEATURES + item_cols

train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

X_train, y_train = train[FEATURES_FULL], train[TARGET]
X_val,   y_val   = val[FEATURES_FULL],   val[TARGET]
X_test,  y_test  = test[FEATURES_FULL],  test[TARGET]

model = HistGradientBoostingRegressor(
    max_depth=6, learning_rate=0.05, max_iter=300,
    early_stopping=True, validation_fraction=0.15, random_state=42
)
model.fit(X_train, y_train)

for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    pred = model.predict(X)
    rmse = mean_squared_error(y, pred) ** 0.5
    mae = mean_absolute_error(y, pred)
    print(f"[Tree expert] {name}: RMSE={rmse:.3f}  MAE={mae:.3f}")

# Save predictions on val+test (what we need downstream for the gate and ablation)
full_eval = pd.concat([val, test]).copy()
full_eval["pred_tree"] = model.predict(full_eval[FEATURES_FULL])

# id was one-hot encoded, so recover original id from the dummy columns for saving
id_recovered = full_eval[item_cols].idxmax(axis=1).str.replace("idflag_", "", regex=False)
out = full_eval[["date", "sales", "split", "pred_tree", "volatile_regime"]].copy()
out["id"] = id_recovered.values
out.to_csv("outputs/tree_predictions.csv", index=False)
print("\nSaved -> outputs/tree_predictions.csv")
print(out.head())
