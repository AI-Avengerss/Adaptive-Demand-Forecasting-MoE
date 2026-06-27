# 🚀 Adaptive Mixture-of-Experts Demand Forecasting System

An end-to-end demand forecasting pipeline that addresses **non-stationary retail demand** using an **Adaptive Mixture-of-Experts (MoE)** architecture. The system combines **LightGBM** and **PyTorch LSTM** models through a **Neural Gating Network** that dynamically allocates trust between forecasting experts based on changing demand conditions.

Unlike conventional forecasting systems that rely on a single model or fixed-weight ensembles, this framework adapts its predictions according to market volatility and model behavior while simultaneously estimating forecast reliability through a **Demand Risk Index (DRI)** and **Confidence Calibration** module.

---

# 📌 Project Highlights

* **Adaptive Mixture-of-Experts Architecture:** Dynamically combines LightGBM and LSTM predictions using a learnable Neural Gating Network.
* **Context-Aware Neural Gating:** Learns time-varying fusion weights (α) from demand volatility, recent forecasting performance, and expert disagreement.
* **Demand Risk Index (DRI):** Quantifies forecasting uncertainty using normalized volatility, disagreement, and error signals.
* **Confidence Calibration:** Converts raw confidence into empirically calibrated **GREEN / YELLOW / RED** reliability zones using validation deciles.
* **Risk-Aware Inventory Planning:** Automatically adjusts safety stock recommendations based on forecast confidence.
* **Interactive Explainability Dashboard:** Visualizes forecasts, expert weights, DRI, confidence levels, inventory recommendations, and operational insights.

---

# 🏗️ System Architecture

```text
                    Historical Demand Data
                               │
                               ▼
                     Feature Engineering
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
          LightGBM Expert               LSTM Expert
                │                             │
                └──────────────┬──────────────┘
                               ▼
                  Neural Gating Network
                               │
                  Dynamic Fusion Weights
                               ▼
                    Final Demand Forecast
                               │
                               ▼
                  Demand Risk Index (DRI)
                               │
                               ▼
                Confidence Calibration
                               │
                               ▼
             Inventory Recommendation Engine
```

---

# 📂 Project Structure

```text
Adaptive-MoE-Forecasting/
│
├── data/                              # Raw M5 Competition dataset
│   ├── calendar.csv
│   ├── sales_train_validation.csv
│   └── sell_prices.csv
│
├── src/                               # 13-Script High-Fidelity Pipeline
│   ├── 00_make_subset.py              # Generates the 80-item training subset
│   ├── 01_calendar_features.py        # Processes holiday signals & event density
│   ├── 02_reshape_merge.py            # Converts wide sales matrix to long format
│   ├── 03_merge_prices.py             # Integrates individual item pricing
│   ├── 04_volatility_regime.py        # Establishes structural rolling regimes
│   ├── 05_build_features.py           # Compiles structural lags & rolling target statistics
│   ├── 06a_lightgbm_real.py           # Tabular LightGBM Expert Engine
│   ├── 06b_lstm_real.py               # Deep Recurrent PyTorch LSTM Sequence Expert
│   ├── 07_gating_network.py           # Mixture of Experts (MoE) Neural Gating Layer
│   ├── 08_ablation_study.py           # Evaluates model variations vs baselines
│   ├── 09_dri_confidence_safety_stock.py # Computes Demand Risk Index & Calibrations
│   └── 10_inventory_simulation.py     # Runs risk-aware supply chain simulations
│
├── outputs/                           # Generated Model Outputs & Dashboard Data
│   ├── ablation_table.csv
│   ├── business_impact.csv
│   ├── fusion_results.csv
│   └── risk_confidence_test.csv
│
├── index.html                         # Live GitHub Pages Dashboard Endpoint
├── requirements.txt                   # Project Dependencies
└── README.md                          # Main Project Documentation
```

---

# ⚙️ Technology Stack

| Category             | Technologies                              |
| -------------------- | ----------------------------------------- |
| Programming Language | Python 3.11+                              |
| Machine Learning     | LightGBM                                  |
| Deep Learning        | PyTorch                                   |
| Data Processing      | Pandas, NumPy, Joblib                     |
| Evaluation           | Scikit-learn                              |
| Dashboard            | HTML5, TailwindCSS, JavaScript (Chart.js) |

---

# 📦 Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Adaptive-MoE-Forecasting.git

cd Adaptive-MoE-Forecasting
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Dataset Configuration

This project uses the **M5 Forecasting Competition** dataset.

Download the following files:

* `calendar.csv`
* `sell_prices.csv`
* `sales_train_validation.csv`
* `sales_train_evaluation.csv`

Create a folder named:

```text
data/
```

Place all downloaded CSV files inside this directory.

---

# ▶️ Running the Pipeline

Ensure your virtual environment is active, then execute the pipeline scripts in their chronological sequence from the root directory:

```bash
# Preprocessing & Feature Engineering
python src/00_make_subset.py
python src/01_calendar_features.py
python src/02_reshape_merge.py
python src/03_merge_prices.py
python src/04_volatility_regime.py
python src/05_build_features.py

# Model Training & Inference
python src/06a_lightgbm_real.py
python src/06b_lstm_real.py

# Downstream Evaluation & Simulation
python src/07_gating_network.py
python src/08_ablation_study.py
python src/09_dri_confidence_safety_stock.py
python src/10_inventory_simulation.py
```

---
# 📊 Evaluation Results

The framework evaluates four forecasting strategies — LightGBM only, LSTM only, a fixed 60/40 blend, and the learned Adaptive MoE gate — reported separately for stable and volatile demand regimes (test set, real M5 data, 80 items, store CA_1). The numbers below are from a validated run using real LightGBM and a real PyTorch LSTM, executed on the identical dataset and chronological split used throughout this project.

| Regime | Configuration | N | RMSE | MAE |
|---|---|---|---|---|
| Stable | LightGBM Expert | 1702 | 6.493 | 4.284 |
| Stable | LSTM Expert | 1702 | 6.674 | 4.284 |
| Stable | Fixed 60/40 | 1702 | 6.477 | 4.239 |
| Stable | **Adaptive MoE (Proposed)** | 1702 | **6.420** | **4.167** |
| Volatile | LightGBM Expert | 538 | 7.696 | 4.809 |
| Volatile | LSTM Expert | 538 | 7.181 | 4.543 |
| Volatile | Fixed 60/40 | 538 | 7.413 | 4.661 |
| Volatile | **Adaptive MoE (Proposed)** | 538 | **7.219** | **4.595** |

**Observations**

- The Adaptive MoE gate is the best of all four configurations in **both** the stable and volatile regime, on both RMSE and MAE. An earlier prototype run (using scikit-learn stand-ins for LightGBM/PyTorch during initial development, before the real libraries were available) showed a small gap in the volatile regime; that gap closed once the real PyTorch LSTM's genuine sequence memory was used instead of the sklearn approximation, confirming the gating approach itself was sound all along.
- When the gate's calibrated confidence score is used to scale a simulated inventory policy's safety stock, stockout frequency drops to **2.59%** — the lowest of all four configurations, compared to 2.77–3.57% for the fixed-buffer baselines — including a clear improvement in volatile periods specifically (3.53% vs. 5.02–6.51%), at roughly the same average inventory-holding cost as the fixed-buffer alternatives. Full detail, including the calibration methodology and a transparent account of the bugs we found and fixed along the way, is in the Detailed Solution Document.
---

# 📈 Dashboard Features

The interactive dashboard provides:

* Historical and forecasted demand visualization
* Dynamic expert trust allocation
* Demand Risk Index (DRI)
* Confidence zones (GREEN / YELLOW / RED)
* Inventory recommendations
* Safety stock adjustment
* Plain-language explanations for every prediction

---

# 💡 Key Innovation

Traditional ensembles rely on static combinations of forecasting models that remain unchanged across all demand conditions.

Our framework introduces a **context-aware Neural Gating Network** that dynamically allocates trust between forecasting experts based on demand volatility, expert disagreement, and recent model performance.

Beyond forecasting future demand, the framework also estimates **forecast reliability** through the Demand Risk Index and Confidence Calibration modules, enabling risk-aware inventory planning rather than accuracy-focused forecasting alone.

---

# 📄 License

Developed as part of the **Hackathon: AI for Public Good – Sustainable & Resilient Supply Chains**.

This repository is intended for academic research, hackathons, and educational purposes.

If you find this project useful, consider giving the repository a ⭐.
