"""
Step 8: The required 4-way ablation study (Objective 3), reported separately
for stable and volatile regimes (Objective 3 explicitly requires this split).

Configurations:
  (a) Tree only      (LightGBM stand-in)
  (b) Sequence only   (LSTM stand-in)
  (c) Fixed 60/40     (0.6 * tree + 0.4 * seq)
  (d) Learned fusion  (gate-driven adaptive weights)

Evaluated on the TEST split only (held-out, never seen by base models or gate).
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("outputs/fusion_results.csv")
test = df[df["split"] == "test"].copy()

def rmse(y, p): return mean_squared_error(y, p) ** 0.5
def mae(y, p):  return mean_absolute_error(y, p)

configs = {
    "(a) Tree only":        "pred_tree",
    "(b) Sequence only":    "pred_seq",
    "(c) Fixed 60/40":      "pred_fixed_60_40",
    "(d) Learned fusion":   "pred_learned_fusion",
}

rows = []
for regime_label, regime_val in [("Stable", 0), ("Volatile", 1)]:
    subset = test[test["volatile_regime"] == regime_val]
    for name, col in configs.items():
        rows.append({
            "Regime": regime_label,
            "Configuration": name,
            "N": len(subset),
            "RMSE": round(rmse(subset["sales"], subset[col]), 3),
            "MAE": round(mae(subset["sales"], subset[col]), 3),
        })

ablation = pd.DataFrame(rows)
print(ablation.to_string(index=False))

ablation.to_csv("outputs/ablation_table.csv", index=False)
print("\nSaved -> outputs/ablation_table.csv")

# Overall (both regimes combined) for reference -- NOT a replacement for the split view,
# the problem statement explicitly requires the split, this is just a sanity-check summary.
print("\n--- Overall (all test rows, for reference only) ---")
for name, col in configs.items():
    print(f"{name}: RMSE={rmse(test['sales'], test[col]):.3f}  MAE={mae(test['sales'], test[col]):.3f}")
