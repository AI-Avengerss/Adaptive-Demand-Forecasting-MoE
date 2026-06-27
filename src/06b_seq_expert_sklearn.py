"""
Step 6b (sandbox stand-in): Train the sequential/deep-learning expert.

SANDBOX NOTE: PyTorch is not installable in this environment (no network access).
This uses sklearn's MLPRegressor over an extended lag window (28 days of lagged
sales) as a practical stand-in for a real LSTM -- it's still a neural network
learning from sequential history, just without a recurrent architecture. The real
PyTorch LSTM script (true sequence model, sliding-window sequences fed through an
nn.LSTM) is provided separately as 06b_lstm_real.py to run on a machine with
PyTorch installed; it reads the same model_dataset.csv and produces the same
prediction columns, so it's a drop-in replacement.
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("outputs/model_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# Build an extended lag window (1..28 days) per item -- this is what gives the
# "sequence expert" a longer memory than the tree expert's shorter lag set.
LAGS = list(range(1, 29))
g = df.groupby("id")["sales"]
for lag in LAGS:
    col = f"seq_lag_{lag}"
    if col not in df.columns:
        df[col] = g.shift(lag)

seq_cols = [f"seq_lag_{l}" for l in LAGS]

# Drop rows without full 28-day history
df["_rownum"] = df.groupby("id").cumcount()
df = df[df["_rownum"] >= 28].drop(columns="_rownum").reset_index(drop=True)
df[seq_cols] = df[seq_cols].fillna(0)

FEATURES = seq_cols + ["wday", "month", "has_event", "snap_active"]
TARGET = "sales"

train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_val   = scaler.transform(val[FEATURES])
X_test  = scaler.transform(test[FEATURES])
y_train, y_val, y_test = train[TARGET].values, val[TARGET].values, test[TARGET].values

model = MLPRegressor(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
    alpha=1e-3, max_iter=500, early_stopping=True, random_state=42
)
model.fit(X_train, y_train)

for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    pred = np.clip(model.predict(X), 0, None)
    rmse = mean_squared_error(y, pred) ** 0.5
    mae = mean_absolute_error(y, pred)
    print(f"[Sequence expert] {name}: RMSE={rmse:.3f}  MAE={mae:.3f}")

full_eval = pd.concat([val, test]).copy()
X_full = scaler.transform(full_eval[FEATURES])
full_eval["pred_seq"] = np.clip(model.predict(X_full), 0, None)

out = full_eval[["id", "date", "sales", "split", "volatile_regime", "pred_seq"]].copy()
out.to_csv("outputs/seq_predictions.csv", index=False)
print("\nSaved -> outputs/seq_predictions.csv")
print(out.head())
