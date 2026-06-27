"""
Step 10: Business Impact Evaluation -- simulated inventory policy.

For each of the 4 ablation configurations, simulate a day-by-day replenishment
policy and measure Stockout Frequency / Inventory Planning Reliability, per the
Round 1 design.

IMPORTANT design choice: only the Learned Fusion configuration gets the DYNAMIC,
confidence-linked safety stock (since only it has a Confidence Score). The other
three configurations (tree-only, seq-only, fixed 60/40) use a FIXED safety stock
equal to the same base_safety_stock value, with no confidence adjustment -- this
isolates exactly what the dynamic safety stock buys you, rather than confounding
"better forecast" with "smarter safety stock" in the same comparison.

Policy (as specified in the Round 1 document):
  Replenishment triggers when: Predicted Demand + Safety Stock > Current Inventory
  Stockout recorded when:      Actual Demand > Available Inventory

Simulation mechanics: starting inventory = first day's actual demand + base safety
stock (a reasonable starting position). Each day, available inventory = previous
day's leftover inventory. If a replenishment was triggered the prior day, assume
it arrives in time to cover the current day's actual demand exactly up to the
order quantity (a simplifying assumption appropriate for a prototype -- next
iteration would add lead time).
"""
import pandas as pd
import numpy as np

test = pd.read_csv("outputs/risk_confidence_test.csv")
test["date"] = pd.to_datetime(test["date"])
test = test.sort_values(["id", "date"]).reset_index(drop=True)

CONFIGS = {
    "(a) Tree only":      ("pred_tree", "fixed"),
    "(b) Sequence only":  ("pred_seq", "fixed"),
    "(c) Fixed 60/40":    ("pred_fixed_60_40", "fixed"),
    "(d) Learned fusion": ("pred_learned_fusion", "dynamic"),
}

def simulate_item(g, pred_col, stock_mode):
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    inventory = np.zeros(n)
    stockouts = np.zeros(n, dtype=bool)
    orders = np.zeros(n)

    base_ss = g["base_safety_stock"].values
    dyn_ss  = g["safety_stock"].values if stock_mode == "dynamic" else base_ss
    pred = g[pred_col].values
    actual = g["sales"].values

    # Starting inventory: cover first day's demand plus a buffer
    inventory[0] = actual[0] + base_ss[0]

    for t in range(n):
        ss = dyn_ss[t] if stock_mode == "dynamic" else base_ss[t]
        available = inventory[t]

        if actual[t] > available:
            stockouts[t] = True

        # End-of-day leftover, floored at 0 (can't go negative in stock)
        leftover = max(available - actual[t], 0)

        # Replenishment decision for the NEXT day, based on predicted demand + safety
        # stock -- using the safety stock VALUE FOR DAY t+1 (the day being covered),
        # not day t's value. Using "today's" confidence to size a buffer for "tomorrow's"
        # demand was a one-day misalignment that mixed up which day's confidence should
        # drive which day's order -- fixed here to use dyn_ss[t+1] / base_ss[t+1].
        if t + 1 < n:
            ss_next = dyn_ss[t + 1] if stock_mode == "dynamic" else base_ss[t + 1]
            target = pred[t + 1] + ss_next
            if target > leftover:
                order_qty = target - leftover
                orders[t] = order_qty
                inventory[t + 1] = leftover + order_qty
            else:
                inventory[t + 1] = leftover

    g["stockout"] = stockouts
    g["order_qty"] = orders
    g["inventory_held"] = inventory
    return g

results = []
detail_frames = {}
for name, (col, mode) in CONFIGS.items():
    sim = test.groupby("id", group_keys=False)[test.columns.tolist()].apply(
        lambda g: simulate_item(g, col, mode)
    )
    stockout_freq = sim["stockout"].mean()
    reliability = 1 - stockout_freq
    avg_order = sim["order_qty"].mean()
    avg_inventory_held = sim["inventory_held"].mean()
    results.append({
        "Configuration": name,
        "Safety Stock Mode": mode,
        "Stockout Frequency": round(stockout_freq, 4),
        "Inventory Planning Reliability": round(reliability, 4),
        "Avg Order Qty": round(avg_order, 2),
        "Avg Inventory Held": round(avg_inventory_held, 2),
    })
    detail_frames[name] = sim

impact = pd.DataFrame(results)
print(impact.to_string(index=False))

impact.to_csv("outputs/business_impact.csv", index=False)
print("\nSaved -> outputs/business_impact.csv")

# Also break out by regime, matching the same stable/volatile split used elsewhere
print("\n--- Stockout Frequency by regime ---")
for name, (col, mode) in CONFIGS.items():
    sim = detail_frames[name]
    by_regime = sim.groupby("volatile_regime")["stockout"].mean()
    print(f"{name}: stable={by_regime.get(0, float('nan')):.4f}  volatile={by_regime.get(1, float('nan')):.4f}")
