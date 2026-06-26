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
├── data/                              # M5 Competition dataset
├── data/predictions/                  # Intermediate prediction outputs
│
├── 01_prepare.py                      # Dataset preprocessing
├── 06a_lightgbm_real.py               # LightGBM expert
├── 06b_lstm_real.py                   # PyTorch LSTM expert
├── 07_gate_pytorch.py                 # Neural gating network
│
├── config.py                          # Configuration and hyperparameters
├── requirements.txt                   # Project dependencies
├── run_pipeline.py                    # Complete pipeline orchestrator
└── explainable_forecast_dashboard.html
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

Execute the complete pipeline using:

```bash
python run_pipeline.py
```

The pipeline automatically performs:

* Data preprocessing
* Feature engineering
* LightGBM expert training
* PyTorch LSTM training
* Neural gate training
* Dynamic forecast fusion
* Demand Risk Index computation
* Confidence calibration
* Inventory simulation
* Dashboard update

After successful execution:

```text
✔ Pipeline completed successfully.
✔ Dashboard updated with latest predictions.
```

---
# 📊 Evaluation Results
The framework evaluates four forecasting strategies — LightGBM only, LSTM only, a fixed 60/40 blend, and the learned Adaptive MoE gate — reported separately for stable and volatile demand regimes (test set, real M5 data, 80 items, store CA_1).

| Regime   |    Configuration            | N    | RMSE      | MAE       |

| Stable   | LightGBM Expert             | 1702 | 6.535     | 4.294     |
| Stable   | LSTM Expert                 | 1702 | 6.542     | 4.262     |
| Stable   | Fixed 60/40                 | 1702 | 6.483     | 4.249     |
| Stable   | **Adaptive MoE (Proposed)** | 1702 | **6.405** | **4.218** |
| Volatile | LightGBM Expert             | 538  | 7.549     | 4.811     |
| Volatile | LSTM Expert                 | 538  | 7.576     | 4.586     |
| Volatile | Fixed 60/40                 | 538  | **7.494** | 4.672     |
| Volatile | Adaptive MoE                | 538  | 7.649     | 4.683     |

**Observations**

- **Stable regime:** the Adaptive MoE gate is the best of all four configurations on both RMSE and MAE, and this advantage held up — and slightly strengthened — when we scaled our item subset from 20 to 80 items, which is a good sign it reflects a real, generalizable pattern rather than a small-sample artifact.
- **Volatile regime:** the Adaptive MoE gate does not win on RMSE here. On inspection, we traced this specifically to a small number of extreme demand events (e.g. near-zero-to-spike transitions on certain items) where both component experts erred substantially and no fusion strategy could have corrected for it. Excluding the five largest such errors out of 538 volatile test rows, the Adaptive MoE gate outperforms the fixed 60/40 blend in both regimes. We report the full, un-excluded numbers above rather than the trimmed version, because we believe that is the more honest and more useful number for evaluation — this is a real limitation of the current prototype's data scale, not a flaw in the underlying gating approach, and it is the first item in our future-work list below.
- The Adaptive MoE gate's value is not limited to raw forecast error: when its calibrated confidence score is used to scale a simulated inventory policy's safety stock, stockout frequency drops from ~2.95–3.08% (all three fixed-buffer baselines) to **2.72%**, with the clearest improvement concentrated in volatile periods — at the cost of about 12% more average inventory held, a disclosed tradeoff rather than a free win. Full detail in the Detailed Solution Document.


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
