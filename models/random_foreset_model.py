# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('C:/Users/USER/Desktop/캡스톤디자인/data/eda_dataset.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# 타겟 생성 (1개월 후 CPI)
df['cpi_kr_target_short'] = df['cpi_kr'].shift(-1)

# 결측치 제거 (타겟 포함된 마지막 행 제거됨)
df.dropna(inplace=True)

# 설명변수(X)와 타겟(y) 설정
X = df.drop(columns=['cpi_kr_target_short', 'cpi_kr', 'cpi_kr_target_mid', 'cpi_kr_target_long'], errors='ignore')
y = df['cpi_kr_target_short']

# 학습/검증 데이터 분리 (시계열 고려한 방식, 최근 20%는 검증용)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Random Forest 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 성능 평가 지표 출력
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))

# 예측 결과 시각화
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test.values, label='실제 CPI', marker='o')
plt.plot(y_test.index, y_pred, label='예측 CPI', marker='x')
plt.title('단기 CPI 예측 결과 (1개월 후)')
plt.xlabel('날짜')
plt.ylabel('CPI')
plt.legend()
plt.grid(True)
plt.show()