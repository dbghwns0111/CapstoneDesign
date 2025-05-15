import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    # 파일 존재 여부 확인
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    return df

def preprocess_data(df):
    # 결측치 처리
    df.fillna(method='ffill', inplace=True)

    # Lag 변수 생성
    lag_months = [1, 2, 3]
    for col in df.columns:
        if col not in ['date', 'cpi_kr']:
            for lag in lag_months:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # 이동평균, 변화량 변수
    df['cpi_kr_ma3'] = df['cpi_kr'].rolling(window=3).mean()
    df['cpi_kr_diff1'] = df['cpi_kr'].diff(1)

    # NaN 제거
    df.dropna(inplace=True)

    # 정규화
    feature_cols = [col for col in df.columns if col not in ['date', 'cpi_kr']]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols

if __name__ == '__main__':
    # 현재 파일의 디렉토리 기준 상대 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, '../data/cpi_dataset.csv')
    save_path = os.path.join(BASE_DIR, '../data/preprocessed_cpi.csv')

    # 데이터 로드 및 전처리
    df = load_data(data_path)
    df, feature_cols = preprocess_data(df)

    # 저장
    df.to_csv(save_path, index=False)
    print(f"✅ 전처리 완료. 저장된 파일: {save_path}")
