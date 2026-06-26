# 06b_lstm_real.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import config

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, df, window_size=30):
        self.window_size = window_size
        self.df = df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
        self.sales = self.df['sales'].values.astype(np.float32)
        
        # Linear sequence verification loop
        self.valid_indices = []
        for i in range(window_size, len(self.df)):
            if (self.df['item_id'].iloc[i] == self.df['item_id'].iloc[i - window_size] and 
                self.df['store_id'].iloc[i] == self.df['store_id'].iloc[i - window_size]):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        x = self.sales[target_idx - self.window_size:target_idx].reshape(-1, 1)
        y = self.sales[target_idx].reshape(1)
        return torch.tensor(x), torch.tensor(y), target_idx

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Use the final hidden sequence state

def train_sequence_expert():
    print("[2/3] Preparing PyTorch Sequenced Tensor Windows...")
    df = pd.read_csv(config.RAW_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    train_df = df[df['date'] < config.TEST_START_DATE].reset_index(drop=True)
    test_df = df[df['date'] >= config.TEST_START_DATE].reset_index(drop=True)
    
    train_ds = TimeSeriesWindowDataset(train_df, window_size=config.WINDOW)
    test_ds = TimeSeriesWindowDataset(test_df, window_size=config.WINDOW)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Train Iterations
    model.train()
    for epoch in range(15): # Optimized for prototype execution speed
        for x_b, y_b, _ in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            
    # Evaluation Loop
    model.eval()
    preds_map = {}
    with torch.no_grad():
        for x_b, _, idx_b in test_loader:
            x_b = x_b.to(device)
            outputs = model(x_b).cpu().numpy()
            for idx, pred_val in zip(idx_b.numpy(), outputs):
                preds_map[idx] = max(0.0, float(pred_val[0])) # Enforce zero-bound demand restriction
                
    test_df['pred_seq'] = np.nan
    for idx, val in preds_map.items():
        test_df.loc[idx, 'pred_seq'] = val
        
    out_df = test_df.dropna(subset=['pred_seq']).reset_index(drop=True)
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "lstm.pth"))
    out_df[['date', 'item_id', 'store_id', 'sales', 'pred_seq']].to_csv(config.SEQ_PRED_PATH, index=False)
    print(f"|-- Sequence inferences cached -> {config.SEQ_PRED_PATH}")

if __name__ == "__main__":
    train_sequence_expert()