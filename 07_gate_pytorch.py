# 07_gate_pytorch.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import config

class MixtureOfExpertsGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)

def train_gating_layer():
    print("[3/3] Initializing Neural Gating Model Training (Mixture-of-Experts Error Minimization Strategy)...")
    
    # Load and align tabular predictions
    t_df = pd.read_csv(config.TREE_PRED_PATH)
    s_df = pd.read_csv(config.SEQ_PRED_PATH)
    m_df = pd.merge(t_df, s_df, on=['date', 'item_id', 'store_id', 'sales']).sort_values('date').reset_index(drop=True)
    
    # Engineer Gating Context Input Vectors (Volatility, Trend, Errors)
    m_df['error_tree'] = (m_df['pred_tree'] - m_df['sales']).abs()
    m_df['rolling_vol'] = m_df['sales'].rolling(7).std().fillna(0)
    m_df['rolling_trend'] = (m_df['sales'].rolling(3).mean() - m_df['sales'].rolling(14).mean()).fillna(0)
    
    # Normalize features for neural network stability
    features = np.stack([m_df['error_tree'].values, m_df['rolling_vol'].values, m_df['rolling_trend'].values], axis=1)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    
    X = torch.tensor(features, dtype=torch.float32)
    p_tree = torch.tensor(m_df['pred_tree'].values, dtype=torch.float32).unsqueeze(1)
    p_seq = torch.tensor(m_df['pred_seq'].values, dtype=torch.float32).unsqueeze(1)
    y_true = torch.tensor(m_df['sales'].values, dtype=torch.float32).unsqueeze(1)
    
    gate = MixtureOfExpertsGate()
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.005)
    
    # Train using standard Custom Backpropagation Layer
    gate.train()
    for epoch in range(30):
        optimizer.zero_grad()
        alphas = gate(X) # Shape: [N, 2]
        a_tree = alphas[:, 0].unsqueeze(1)
        a_seq = alphas[:, 1].unsqueeze(1)
        
        # Softmax Constraint Enforced Differentiable Forecast Model
        y_pred_fusion = (a_tree * p_tree) + (a_seq * p_seq)
        loss = torch.mean((y_pred_fusion - y_true) ** 2) # Direct Optimization of Mixture Loss
        loss.backward()
        optimizer.step()
        
    # Generate final predictions
    gate.eval()
    with torch.no_grad():
        final_alphas = gate(X).numpy()
        
    m_df['alpha_tree_gate'] = final_alphas[:, 0]
    m_df['pred_learned_fusion'] = (m_df['alpha_tree_gate'] * m_df['pred_tree']) + ((1 - m_df['alpha_tree_gate']) * m_df['pred_seq'])
    m_df['pred_fixed_60_40'] = (0.60 * m_df['pred_tree']) + (0.40 * m_df['pred_seq'])
    
    # Derive Demand Risk Index (DRI) directly from alpha dynamics
    m_df['dri'] = (m_df['alpha_tree_gate'] - 0.5).abs() * 2 * 0.4 + 0.2
    m_df['calibrated_confidence'] = 100 * (1.0 - m_df['dri'])
    m_df['zone'] = np.where(m_df['calibrated_confidence'] > 65, "GREEN", np.where(m_df['calibrated_confidence'] > 63, "YELLOW", "RED"))
    
    # Supply Chain Metrics
    m_df['safety_stock'] = m_df['pred_learned_fusion'] * 0.2 * (1.5 if m_df['zone'].all() == "RED" else 1.0)
    m_df['base_safety_stock'] = m_df['pred_learned_fusion'] * 0.15
    m_df['volatile_regime'] = np.where(m_df['rolling_vol'] > m_df['rolling_vol'].median(), 1, 0)
    
    torch.save(gate.state_dict(), os.path.join(config.MODEL_DIR, "gate.pth"))
    m_df.to_csv(config.FUSION_PRED_PATH, index=False)
    print(f"|-- Gating model optimization finished -> {config.FUSION_PRED_PATH}")

if __name__ == "__main__":
    train_gating_layer()