import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# CUDA 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 현재 장치: {device}")

# 설정
seq_len = 24
start_eval_date = "2010-01-01"

df = pd.read_csv("C:/캡스톤디자인/CapstoneDesign/data/상관계수 제거버전/석유류_품목.csv", encoding="cp949")
df["날짜"] = pd.to_datetime(df["날짜"])
df.set_index("날짜", inplace=True)
item = "휘발유"
data = df[item].dropna()

# YoY 계산
yoy = data.pct_change(12) * 100
yoy.dropna(inplace=True)
scaler = RobustScaler()
yoy_scaled = scaler.fit_transform(yoy.values.reshape(-1, 1)).flatten()

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# sliding window 기반 예측
y_true, y_pred, pred_dates = [], [], []

# 평가 시작 인덱스 계산
eval_start_idx = yoy.index.get_loc(pd.to_datetime(start_eval_date))

for i in range(eval_start_idx, len(yoy_scaled) - 1):
    past_seq = yoy_scaled[i - seq_len:i]
    if len(past_seq) < seq_len:
        continue

    X_train = []
    y_train = []
    for j in range(0, i - seq_len):
        X_train.append(yoy_scaled[j:j + seq_len])
        y_train.append(yoy_scaled[j + seq_len])

    if len(X_train) == 0:
        continue

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

    # 모델 초기화 및 학습
    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(20):
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 예측
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(past_seq, dtype=torch.float32).reshape(1, seq_len, 1).to(device)
        pred = model(input_seq).cpu().item()
        y_pred.append(scaler.inverse_transform([[pred]])[0, 0])
        y_true.append(data.iloc[i + 1] / data.iloc[i + 1 - 12] * 100 - 100)
        pred_dates.append(data.index[i + 1])

# 성능 출력
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
print(f"\n검증 시작 시점: {start_eval_date} ~ {pred_dates[-1].strftime('%Y-%m')} 까지")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(pred_dates, y_true, label="실제 YoY")
plt.plot(pred_dates, y_pred, label="예측 YoY")
plt.title(f"{item} CPI YoY 예측 vs 실제")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 예측 결과를 CSV 파일로 저장
pred_df = pd.DataFrame({"날짜": pred_dates, "실제 YoY": y_true, "예측 YoY": y_pred})
pred_df.set_index("날짜", inplace=True)
pred_df.to_csv("C:/캡스톤디자인/CapstoneDesign/results/석유류_품목_예측결과.csv", encoding="cp949")