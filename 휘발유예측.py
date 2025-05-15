import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import random
import os


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("C:/Users/USER/desktop/캡스톤디자인/data/상관계수 제거버전/석유류_품목.csv", encoding="cp949")
df["날짜"] = pd.to_datetime(df["날짜"])
df.set_index("날짜", inplace=True)

item = "휘발유"
data = df[item].dropna()

# YoY 계산
yoy = data.pct_change(12) * 100
yoy.dropna(inplace=True)


scaler = RobustScaler()
yoy_scaled = scaler.fit_transform(yoy.values.reshape(-1, 1))


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# 하이퍼파라미터
seq_len = 24
n_val = 12
n_forecast = 12


X_all, y_all = create_sequences(yoy_scaled, seq_len)
X_all = X_all.reshape(X_all.shape[0], seq_len, 1)


X_train = X_all[:-n_val]
y_train = y_all[:-n_val]
X_val = X_all[-n_val:]
y_val = y_all[-n_val:]


def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

# 1. 검증용 모델 학습
model_val = build_model((seq_len, 1))
model_val.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_val, y_val))


val_preds_scaled = model_val.predict(X_val)
val_preds = scaler.inverse_transform(val_preds_scaled).flatten()
y_val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

# 성능 출력
mae = mean_absolute_error(y_val_true, val_preds)
rmse = np.sqrt(mean_squared_error(y_val_true, val_preds))
r2 = r2_score(y_val_true, val_preds)

print("\n 검증 성능 (2024-03 ~ 2025-02)")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

# 날짜 생성
val_dates = data.index[-(n_val + n_forecast):-n_forecast]

#  전체 데이터 재학습 (검증 포함)
model_full = build_model((seq_len, 1))
model_full.fit(X_all, y_all, epochs=50, batch_size=16, verbose=0)

# 예측 (2025-03 ~ 2026-02)
last_input = yoy_scaled[-seq_len:].reshape(1, seq_len, 1)
future_preds_scaled = []
for _ in range(n_forecast):
    pred = model_full.predict(last_input, verbose=0)[0, 0]
    future_preds_scaled.append(pred)
    last_input = np.append(last_input[:, 1:, :], [[[pred]]], axis=1)

future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
future_dates = pd.date_range(start="2025-03-01", periods=n_forecast, freq='MS')

forecast_df = pd.DataFrame({
    "날짜": future_dates,
    "예측_CPI_YoY": future_preds
})
forecast_df.to_csv("cpi 예측 데이터 예시.csv", index=False, encoding="cp949")
print(" 예측 결과 저장됨")
# 시각화
plt.figure(figsize=(12, 6))
plt.plot(val_dates, y_val_true, label="실제 YoY")
plt.plot(val_dates, val_preds, label="검증 예측")
plt.plot(future_dates, future_preds, label="미래 예측 (2025-03~)", linestyle='--')
plt.title(f"{item} CPI YoY 예측")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

