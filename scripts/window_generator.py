# scripts/window_generator.py

import numpy as np
import pandas as pd

def create_sliding_window(df: pd.DataFrame, input_days=310, forecast_days=31):
    #def create_sliding_window(df: pd.DataFrame, input_days=310, forecast_days=31, target_column='Total_CPI'):
    """
    시계열 슬라이딩 윈도우 구성 함수

    Parameters:
        df (pd.DataFrame): 일별 정규화된 CPI 데이터
        input_days (int): 입력 구간 (기본 10개월 = 310일)
        forecast_days (int): 예측 구간 (기본 1개월 = 31일)
        target_column (str): 예측할 타겟 컬럼 이름

    Returns:
        X (np.ndarray): (샘플 수, input_days, 변수 수)
        y (np.ndarray): (샘플 수, forecast_days)
    """
    data = df.copy().values
    #target_idx = df.columns.get_loc(target_column)

    X, y = [], []

    for i in range(len(data) - input_days - forecast_days):
        X.append(data[i:i + input_days])
        #y.append(data[i + input_days:i + input_days + forecast_days, target_idx])
        y.append(data[i + input_days:i + input_days + forecast_days, 0])

    return np.array(X), np.array(y)
