"""
Step 6a (REAL): Train the tree-based expert using actual LightGBM.

This is a drop-in replacement for 06a_tree_expert_sklearn.py, using the exact
same features, the same 80-item dataset, and the same train/val/test split
(produced by 05_build_features.py) -- only the model itself differs.

Requires: pip install lightgbm
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("outputs/model_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

FEATURES = [
    "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_7", "sales_lag_14",
    "roll_mean_7_lag1", "roll_std_7_lag1", "roll_mean_14_lag1", "roll_std_14_lag1",
    "wday", "month", "has_event", "snap_active", "sell_price", "price_pct_change",
]
TARGET = "sales"

# Same one-hot item encoding as the sklearn version, for a fair comparison
df = pd.get_dummies(df, columns=["id"], prefix="idflag")
item_cols = [c for c in df.columns if c.startswith("idflag_")]
FEATURES_FULL = FEATURES + item_cols

train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

X_train, y_train = train[FEATURES_FULL], train[TARGET]
X_val,   y_val   = val[FEATURES_FULL],   val[TARGET]
X_test,  y_test  = test[FEATURES_FULL],  test[TARGET]

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)

for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    pred = model.predict(X)
    rmse = mean_squared_error(y, pred) ** 0.5
    mae = mean_absolute_error(y, pred)
    print(f"[LightGBM Tree expert] {name}: RMSE={rmse:.3f}  MAE={mae:.3f}")

full_eval = pd.concat([val, test]).copy()
full_eval["pred_tree"] = model.predict(full_eval[FEATURES_FULL])

id_recovered = full_eval[item_cols].idxmax(axis=1).str.replace("idflag_", "", regex=False)
out = full_eval[["date", "sales", "split", "pred_tree", "volatile_regime"]].copy()
out["id"] = id_recovered.values
out.to_csv("outputs/tree_predictions.csv", index=False)
print("\nSaved -> outputs/tree_predictions.csv")
print(out.head())
