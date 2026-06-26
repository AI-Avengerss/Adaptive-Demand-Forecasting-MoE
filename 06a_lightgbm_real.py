# 06a_lightgbm_real.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config

def build_tabular_features(df_path):
    df = pd.read_csv(df_path)
    df = df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
    
    # Generate lag structures
    for lag in [1, 2, 7, 14]:
        df[f'lag_{lag}'] = df.groupby(['store_id', 'item_id'])['sales'].shift(lag)
        
    # Generate rolling statistics from lookback window
    for w in [7, 30]:
        df[f'rolling_mean_{w}'] = df.groupby(['store_id', 'item_id'])['lag_1'].transform(lambda x: x.rolling(w).mean())
        df[f'rolling_std_{w}'] = df.groupby(['store_id', 'item_id'])['lag_1'].transform(lambda x: x.rolling(w).std())
        
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Explicit native optimization categorical data types
    for col in ['item_id', 'store_id']:
        df[col] = df[col].astype('category')
        
    return df.dropna().reset_index(drop=True)

def train_tree_expert():
    print("[1/3] Engineering LightGBM tabular feature matrices...")
    df = build_tabular_features(config.RAW_DATA_PATH)
    
    # Train/Test Temporal Splitting
    train_df = df[df['date'] < config.TEST_START_DATE].reset_index(drop=True)
    test_df = df[df['date'] >= config.TEST_START_DATE].reset_index(drop=True)
    
    features = [c for c in df.columns if c not in ['date', 'sales']]
    
    # Model definition using LightGBM Scikit-Learn Wrapper
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(train_df[features], train_df['sales'])
    
    # Validation Logging
    test_df['pred_tree'] = model.predict(test_df[features])
    rmse = np.sqrt(mean_squared_error(test_df['sales'], test_df['pred_tree']))
    print(f"|-- LightGBM Tree Expert Evaluated. Test Set RMSE: {rmse:.4f}")
    
    # Save Artifacts
    joblib.dump(model, os.path.join(config.MODEL_DIR, "lightgbm.pkl"))
    test_df[['date', 'item_id', 'store_id', 'sales', 'pred_tree']].to_csv(config.TREE_PRED_PATH, index=False)
    print(f"|-- Tree inferences cached -> {config.TREE_PRED_PATH}")

if __name__ == "__main__":
    train_tree_expert()