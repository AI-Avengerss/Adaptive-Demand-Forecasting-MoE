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

The framework evaluates four forecasting strategies:

| Scenario        | Model Strategy              |      RMSE |       MAE |
| --------------- | --------------------------- | --------: | --------: |
| Stable Demand   | **Adaptive MoE (Proposed)** | **6.405** | **4.218** |
| Stable Demand   | LightGBM Expert             |     6.535 |     4.296 |
| Stable Demand   | LSTM Expert                 |     6.542 |     4.312 |
| Volatile Demand | **Fixed 60/40 Fallback**    | **7.494** |         — |
| Volatile Demand | Adaptive MoE                |     7.649 |         — |

**Observations**

* Under stable demand conditions, the Adaptive Mixture-of-Experts model achieves the lowest forecasting error.
* During highly volatile demand regimes, the fixed fallback strategy provides more stable operational behavior, making it suitable as a risk-management mechanism.

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
