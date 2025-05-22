# scripts/data_augment.py

import pandas as pd

def interpolate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    월간 CPI 데이터를 일별로 선형 보간하는 함수

    Parameters:
        df (pd.DataFrame): 월별 CPI 데이터, datetime index 필요

    Returns:
        pd.DataFrame: 일별로 보간된 CPI 데이터
    """
    # 인덱스를 월초로 정렬
    df.index = pd.to_datetime(df.index)
    df = df.resample('MS').first()  # 월 초 기준 정렬

    # 일별로 인덱스 생성
    df_daily = df.resample('D').interpolate(method='linear')
    return df_daily
