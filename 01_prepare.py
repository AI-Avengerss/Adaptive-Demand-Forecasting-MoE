# 01_prepare.py
import pandas as pd
import os
import config

def prepare_m5_data():
    print("[0/3] Processing raw M5 dataset files...")
    
    # Paths to your raw files
    sales_path = os.path.join(config.DATA_DIR, "sales_train_evaluation.csv")
    calendar_path = os.path.join(config.DATA_DIR, "calendar.csv")
    
    if not os.path.exists(sales_path):
        # Fallback to validation file if evaluation isn't present
        sales_path = os.path.join(config.DATA_DIR, "sales_train_validation.csv")

    # 1. Load data and filter for the target item/store on your dashboard
    print("|-- Loading sales records...")
    sales_df = pd.read_csv(sales_path)
    filtered_sales = sales_df[
        (sales_df['item_id'] == 'FOODS_3_607') & 
        (sales_df['store_id'] == 'CA_1')
    ]
    
    # 2. Reshape wide days (d_1, d_2...) into a long row format
    print("|-- Transforming time-series matrix layout...")
    day_cols = [c for c in filtered_sales.columns if c.startswith('d_')]
    id_cols = ['item_id', 'store_id']
    
    melted_df = pd.melt(
        filtered_sales, 
        id_vars=id_cols, 
        value_vars=day_cols, 
        var_name='d', 
        value_name='sales'
    )
    
    # 3. Join with calendar to get real dates
    print("|-- Aligning calendar timelines...")
    calendar_df = pd.read_csv(calendar_path)[['date', 'd']]
    final_df = pd.merge(melted_df, calendar_df, on='d')
    
    # Clean up structure
    final_df = final_df[['date', 'item_id', 'store_id', 'sales']]
    
    # Save directly to the path the experts are looking for
    final_df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"|-- Success! Created clean dataset at: {config.RAW_DATA_PATH}")

if __name__ == "__main__":
    prepare_m5_data()