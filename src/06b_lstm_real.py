"""
Step 6b (REAL): Train the deep learning expert using an actual PyTorch LSTM.

This is a drop-in replacement for 06b_seq_expert_sklearn.py. Unlike the sklearn
stand-in (an MLP fed a flattened 28-day lag window), this uses a genuine
recurrent nn.LSTM that processes the sequence step-by-step, which is what the
problem statement and our Detailed Solution Document describe as the deep
learning component.

Requires: pip install torch
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 28  # same lookback window as the sklearn stand-in, for a fair comparison

df = pd.read_csv("outputs/model_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# Scale sales per item isn't needed here since the gate/ablation operate on raw
# units; we scale only as model input and invert before reporting predictions.
scaler = StandardScaler()
df["sales_scaled"] = scaler.fit_transform(df[["sales"]])

def build_sequences(group):
    """For each day t (with at least SEQ_LEN prior days), build a (SEQ_LEN, 1)
    input sequence of scaled sales values ending at t-1, predicting day t."""
    group = group.sort_values("date").reset_index(drop=True)
    seqs, targets, meta = [], [], []
    vals = group["sales_scaled"].values
    raw_vals = group["sales"].values
    for i in range(SEQ_LEN, len(group)):
        seqs.append(vals[i - SEQ_LEN:i])
        targets.append(raw_vals[i])
        meta.append(i)
    return seqs, targets, meta

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # last timestep's hidden state
        return self.fc(last).squeeze(-1)

print("Building per-item sequences (this constructs sliding windows for every item)...")
all_X, all_y, all_meta = [], [], []
for item_id, group in df.groupby("id"):
    seqs, targets, idxs = build_sequences(group)
    for s, t, idx in zip(seqs, targets, idxs):
        all_X.append(s)
        all_y.append(t)
        all_meta.append((item_id, group.iloc[idx]["date"], group.iloc[idx]["split"],
                          group.iloc[idx]["volatile_regime"]))

X = np.array(all_X, dtype=np.float32).reshape(-1, SEQ_LEN, 1)
y = np.array(all_y, dtype=np.float32)
meta_df = pd.DataFrame(all_meta, columns=["id", "date", "split", "volatile_regime"])

train_mask = (meta_df["split"] == "train").values
val_mask   = (meta_df["split"] == "val").values
test_mask  = (meta_df["split"] == "test").values

X_train, y_train = torch.tensor(X[train_mask]), torch.tensor(y[train_mask])
X_val,   y_val   = torch.tensor(X[val_mask]),   torch.tensor(y[val_mask])
X_test,  y_test  = torch.tensor(X[test_mask]),  torch.tensor(y[test_mask])

model = LSTMForecaster(hidden_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("Training LSTM...")
best_val_loss = float("inf")
patience, patience_counter = 10, 0
EPOCHS = 100
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_train.shape[0])
    epoch_loss = 0.0
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= X_train.shape[0]

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred, y_val).item()

    if epoch % 10 == 0:
        print(f"  epoch {epoch}: train_loss={epoch_loss:.3f}  val_loss={val_loss:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    for name, X_t, y_t in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = model(X_t).numpy()
        pred = np.clip(pred, 0, None)
        rmse = mean_squared_error(y_t.numpy(), pred) ** 0.5
        mae = mean_absolute_error(y_t.numpy(), pred)
        print(f"[PyTorch LSTM expert] {name}: RMSE={rmse:.3f}  MAE={mae:.3f}")

    full_mask = val_mask | test_mask
    X_full = torch.tensor(X[full_mask])
    pred_full = np.clip(model(X_full).numpy(), 0, None)

out = meta_df[full_mask].copy()
out["pred_seq"] = pred_full
out = out[["id", "date", "split", "volatile_regime"]].copy()
out["pred_seq"] = pred_full
# Re-attach actual sales for downstream consistency with the sklearn version's output format
sales_lookup = df.set_index(["id", "date"])["sales"]
out["sales"] = out.set_index(["id", "date"]).index.map(sales_lookup)
out = out[["id", "date", "sales", "split", "volatile_regime", "pred_seq"]]

out.to_csv("outputs/seq_predictions.csv", index=False)
print("\nSaved -> outputs/seq_predictions.csv")
print(out.head())
