"""
Step 7: Adaptive gating network.

SANDBOX NOTE: same constraint as the base experts -- this uses sklearn's
MLPClassifier-style setup via MLPRegressor + manual softmax, as a stand-in for
the real PyTorch 3-layer MLP + Softmax gate described in the submission. The
real PyTorch gating script is provided separately (06b_lstm_real.py /
07_gate_real.py) to run on your machine.

Gate inputs (context features the BASE MODELS do not see directly as a routing
signal -- this is what makes it "beyond a stacked meta-learner"):
  - rolling_cv_14      (volatility)
  - trend_accel        (trend acceleration)
  - has_event          (calendar context)
  - snap_active        (calendar context)
  - disagreement       (|pred_tree - pred_seq|, normalized)

Gate output: alpha_tree, alpha_seq (softmax, sum to 1) -- the fusion weights.

Training target for the gate: rather than hand-picking weights, the gate is
trained to minimize the fused prediction's error, i.e. it learns, from data,
which expert to favor in which context. Since sklearn has no native "softmax
output trained end-to-end on downstream loss" regressor, we approximate this
with a practical two-step approach used commonly as a lightweight gating
heuristic-with-learning:
  1. Train a small MLP regressor to predict the OPTIMAL alpha_tree for each
     row (defined as the alpha that would have minimized that row's fused
     error, computed in hindsight on the val set).
  2. Apply softmax-style normalization to keep outputs in [0,1] and use the
     learned mapping at inference (test set) time -- i.e. the gate generalizes
     the hindsight-optimal weighting pattern to unseen data, rather than
     memorizing fixed weights.
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

regime_df = pd.read_csv("outputs/sales_with_regime.csv")[
    ["id", "date", "rolling_cv_14", "trend_accel", "has_event", "snap_active", "volatile_regime"]
]
regime_df["date"] = pd.to_datetime(regime_df["date"])

tree = pd.read_csv("outputs/tree_predictions.csv")
seq  = pd.read_csv("outputs/seq_predictions.csv")
tree["date"] = pd.to_datetime(tree["date"])
seq["date"] = pd.to_datetime(seq["date"])

df = tree.merge(seq[["id", "date", "pred_seq"]], on=["id", "date"])
df = df.drop(columns=["volatile_regime"])  # will re-merge a clean copy from regime_df below
df = df.merge(regime_df, on=["id", "date"], how="left")
df = df.fillna(0)

# Disagreement signal, normalized by the 95th percentile (consistent with the DRI design)
df["disagreement_raw"] = (df["pred_tree"] - df["pred_seq"]).abs()
disagreement_95 = df["disagreement_raw"].quantile(0.95)
df["disagreement_norm"] = (df["disagreement_raw"] / max(disagreement_95, 1e-6)).clip(0, 1)

# Hindsight-optimal alpha_tree per row: the weight on the tree model that would
# have produced the lowest error for THIS row, searched over a fine grid.
# fused = alpha * pred_tree + (1-alpha) * pred_seq
grid = np.linspace(0, 1, 21)

# (Per-row grid search replaced below by a locally-smoothed rolling-window version --
# the naive per-row version was too noisy to be a learnable target, see note below.)

# Hindsight-optimal alpha_tree per row degenerates to a near-binary target (whichever
# single model happened to be closer for that exact row), which is too noisy for a
# smooth regression target -- the per-row "optimal" is dominated by noise, not signal.
# Instead, compute a LOCALLY SMOOTHED optimal alpha: for each row, find the alpha that
# minimizes total error over a rolling 7-day window centered on that row (per item).
# This reflects a genuine regime-level pattern rather than single-point noise, and is
# the target the gate can realistically learn to generalize.
# Locally-smoothed optimal alpha (per item, 7-day window centered on each row):
def rolling_optimal_alpha(group):
    group = group.sort_values("date").reset_index(drop=True)
    n = len(group)
    alphas = np.full(n, 0.5)
    pred_tree = group["pred_tree"].values
    pred_seq = group["pred_seq"].values
    sales = group["sales"].values
    half = 3  # 7-day window
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        pt, ps, s = pred_tree[lo:hi], pred_seq[lo:hi], sales[lo:hi]
        errs = np.abs(np.outer(grid, pt) + np.outer(1 - grid, ps) - s).mean(axis=1)
        alphas[i] = grid[np.argmin(errs)]
    group["alpha_optimal"] = alphas
    return group

df = df.groupby("id", group_keys=False)[df.columns.tolist()].apply(rolling_optimal_alpha)
df = df.reset_index(drop=True)

GATE_FEATURES = ["rolling_cv_14", "trend_accel", "has_event", "snap_active", "disagreement_norm"]

train_mask = df["split"] == "val"   # gate learns from val (held-out from base model training)
test_mask  = df["split"] == "test"  # gate is evaluated on test (never seen by base models or gate)

X_train_raw = df.loc[train_mask, GATE_FEATURES].values
X_test_raw  = df.loc[test_mask,  GATE_FEATURES].values
pt_train = df.loc[train_mask, "pred_tree"].values
ps_train = df.loc[train_mask, "pred_seq"].values
y_train  = df.loc[train_mask, "sales"].values
pt_test  = df.loc[test_mask, "pred_tree"].values
ps_test  = df.loc[test_mask, "pred_seq"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# --- Small 2-layer MLP gate, trained END-TO-END to minimize the FUSED prediction's
# error directly (not a noisy proxy target). This is a deliberate fix: an earlier
# version trained the gate to match a per-row/locally-smoothed "optimal alpha" label,
# but with only two competing experts that label is close to binary and too noisy to
# be a good regression target. Training directly on downstream fusion error is the
# correct approach and is also what the real PyTorch gate (07_gate_real.py) does via
# backprop through the fused loss.
rng = np.random.default_rng(42)
n_features = X_train.shape[1]
hidden = 8

W1 = rng.normal(0, 0.3, (n_features, hidden))
b1 = np.zeros(hidden)
W2 = rng.normal(0, 0.3, hidden)
b2 = 0.0

def forward(X, W1, b1, W2, b2):
    h = np.tanh(X @ W1 + b1)
    z = h @ W2 + b2
    alpha = 1 / (1 + np.exp(-z))  # sigmoid -> alpha_tree in (0,1); alpha_seq = 1-alpha
    return h, alpha

lr = 0.1
n_epochs = 3000
n = X_train.shape[0]
l2_lambda = 0.05  # L2 regularization: penalizes large weights, which is what was letting
                  # the gate swing to near-0/near-1 confidence on the ~50 sparse,
                  # high-disagreement training rows rather than staying moderate when
                  # evidence is thin. This is the actual fix for the volatile-regime
                  # regression identified by inspecting individual worst-case rows.

for epoch in range(n_epochs):
    h, alpha = forward(X_train, W1, b1, W2, b2)
    fused = alpha * pt_train + (1 - alpha) * ps_train
    err = fused - y_train  # d(loss)/d(fused) for MSE loss

    # Backprop through: loss -> fused -> alpha (sigmoid) -> z -> (W2,b2) -> h (tanh) -> (W1,b1)
    d_fused = 2 * err / n
    d_alpha = d_fused * (pt_train - ps_train)
    d_z = d_alpha * alpha * (1 - alpha)
    dW2 = h.T @ d_z + l2_lambda * W2
    db2 = d_z.sum()
    d_h = np.outer(d_z, W2)
    d_h_pre = d_h * (1 - h ** 2)
    dW1 = X_train.T @ d_h_pre + l2_lambda * W1
    db1 = d_h_pre.sum(axis=0)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if epoch % 500 == 0:
        loss = np.mean(err ** 2)
        print(f"  epoch {epoch}: fused MSE = {loss:.3f}")

_, alpha_train = forward(X_train, W1, b1, W2, b2)
_, alpha_test  = forward(X_test,  W1, b1, W2, b2)

df.loc[train_mask, "alpha_tree_gate"] = alpha_train
df.loc[test_mask,  "alpha_tree_gate"] = alpha_test
df["alpha_seq_gate"] = 1 - df["alpha_tree_gate"]

df["pred_learned_fusion"] = df["alpha_tree_gate"] * df["pred_tree"] + df["alpha_seq_gate"] * df["pred_seq"]
df["pred_fixed_60_40"] = 0.6 * df["pred_tree"] + 0.4 * df["pred_seq"]

print("\n=== Gate weight summary (test split) ===")
print(df.loc[test_mask, "alpha_tree_gate"].describe())
print()
print("Mean alpha_tree by regime (test split):")
print(df.loc[test_mask].groupby("volatile_regime")["alpha_tree_gate"].mean())

df.to_csv("outputs/fusion_results.csv", index=False)
print("\nSaved -> outputs/fusion_results.csv")
