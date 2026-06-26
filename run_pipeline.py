# run_pipeline.py
import subprocess
import json
import os
import pandas as pd
import config

def run_ml_pipeline():
    print("=================================================================")
    print("      LAUNCHING EXPLAINABLE FORECAST PIPELINE SUB-SYSTEM         ")
    print("=================================================================")
    
    # 1. Run data preparation FIRST to create the missing raw_m5_data.csv
    print("\n[Step 1/4] Running Data Preparation...")
    subprocess.run(["python", "01_prepare.py"], check=True)
    
    # 2. Run the LightGBM Tree Expert
    print("\n[Step 2/4] Running LightGBM Tree Expert...")
    subprocess.run(["python", "06a_lightgbm_real.py"], check=True)
    
    # 3. Run the PyTorch LSTM Sequence Expert
    print("\n[Step 3/4] Running PyTorch LSTM Expert...")
    subprocess.run(["python", "06b_lstm_real.py"], check=True)
    
    # 4. Run the Neural Gating Layer Optimization
    print("\n[Step 4/4] Running Neural Gating Network...")
    subprocess.run(["python", "07_gate_pytorch.py"], check=True)
    
    print("\n[Processing] Formatting pipeline outputs into Frontend Visualizations JSON...")
    
    # 5. Extract calculations and convert to JSON format
    df = pd.read_csv(config.FUSION_PRED_PATH)
    records = []
    for _, r in df.iterrows():
        records.append({
            "date": str(r['date'])[:10],
            "sales": int(r['sales']),
            "pred_tree": float(r['pred_tree']),
            "pred_seq": float(r['pred_seq']),
            "pred_learned_fusion": float(r['pred_learned_fusion']),
            "pred_fixed_60_40": float(r['pred_fixed_60_40']),
            "alpha_tree_gate": float(r['alpha_tree_gate']),
            "dri": float(r['dri']),
            "calibrated_confidence": float(r['calibrated_confidence']),
            "zone": str(r['zone']),
            "volatile_regime": int(r['volatile_regime']),
            "safety_stock": float(r['safety_stock']),
            "base_safety_stock": float(r['base_safety_stock'])
        })
        
    # Take a 28-day sample to match the dashboard's chart layouts
    json_payload = json.dumps(records[:28], indent=2)
    
    # 6. Read your standalone HTML prototype file
    dashboard_file = "explainable_forecast_dashboard.html"
    if not os.path.exists(dashboard_file):
        print(f"ERROR: Could not find {dashboard_file} in your folder!")
        return
        
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        html_code = f.read()
        
    # 7. Automatically overwrite the static placeholder data inside your HTML file
    import re
    pattern = r"const DATA = \[.*?\];"
    replacement = f"const DATA = {json_payload};"
    modified_html = re.sub(pattern, replacement, html_code, flags=re.DOTALL)
    
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(modified_html)
        
    print(f"\n|-- SUCCESS! Pipeline finished. {dashboard_file} updated with live data.")
    print("=================================================================")

if __name__ == "__main__":
    run_ml_pipeline()