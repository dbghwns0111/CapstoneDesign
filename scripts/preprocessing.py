# scripts/preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_normalize_data(csv_path):
    """
    CSV에서 데이터를 불러오고 MinMaxScaler를 이용해 정규화합니다.
    
    Parameters:
        csv_path (str): 원본 CSV 경로
    
    Returns:
        pd.DataFrame: 정규화된 데이터프레임
        MinMaxScaler: 학습된 스케일러 객체
    """
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df, scaler
