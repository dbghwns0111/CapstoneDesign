import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 현재 경로 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '../data/preprocessed_cpi.csv')
output_path = os.path.join(BASE_DIR, '../results/figures/lstm_prediction.png')

# 데이터 불러오기
df = pd.read_csv(data_path, parse_dates=['date'])

# 타겟 및 특성 정의
target_col = 'cpi_kr'
feature_cols = [col for col in df.columns if col not in ['date', target_col]]

X = df[feature_cols].values
y = df[target_col].values

# 시계열 데이터를 LSTM 입력 형태로 변환
def create_sequences(X, y, window_size=12):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])  # 다음 달 CPI
    return np.array(Xs), np.array(ys)

window_size = 12
X_seq, y_seq = create_sequences(X, y, window_size)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# LSTM 모델 정의
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                    validation_split=0.2, verbose=1)

# 예측
y_pred = model.predict(X_test)

# 결과 시각화
plt.figure(figsize=(10, 4))
# 실제 CPI
plt.plot(df['date'].iloc[-len(y_test):], y_test, label='Actual CPI', color='blue')
# 예측 CPI
plt.plot(df['date'].iloc[-len(y_test):], y_pred, label='Predicted CPI', color='red')
plt.title('LSTM CPI Prediction')
plt.xlabel('time')
plt.ylabel('CPI')
plt.legend()
plt.tight_layout()

# 저장
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.show()
