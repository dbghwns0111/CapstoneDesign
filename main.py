import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================================
# 0. 모듈 로딩
# ================================
try:
    print("🔧 Importing modules...")
    from scripts.preprocessing import load_and_normalize_data
    from scripts.data_augment import interpolate_to_daily
    from scripts.window_generator import create_sliding_window
    from scripts.model import CNNLSTMModel
    print("✅ 모듈 로딩 완료\n")
except Exception as e:
    print(f"❌ 모듈 로딩 실패: {e}")
    exit(1)

# ================================
# 1. 디바이스 설정
# ================================
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 실행 디바이스: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}\n")
except Exception as e:
    print(f"❌ 디바이스 설정 실패: {e}")
    exit(1)

# ================================
# 2. 데이터 정규화
# ================================
try:
    print("📥 Step 1: Load & Normalize Data")
    raw_path = './data/raw/fred_data_2010_2021.csv'
    df_scaled, _ = load_and_normalize_data(raw_path)
    print("✅ 정규화 완료\n")
except Exception as e:
    print(f"❌ 데이터 정규화 실패: {e}")
    exit(1)

# ================================
# 3. 일별 보간
# ================================
try:
    print("📈 Step 2: Interpolate to Daily")
    df_daily = interpolate_to_daily(df_scaled)
    os.makedirs('./data/processed', exist_ok=True)
    df_daily.to_csv('./data/processed/cpi_interpolated.csv')
    print("✅ 일별 보간 및 저장 완료\n")
except Exception as e:
    print(f"❌ 일별 보간 실패: {e}")
    exit(1)

# ================================
# 4. 슬라이딩 윈도우 구성
# ================================
try:
    print("🧱 Step 3: Create Sliding Windows")
    X, y = create_sliding_window(df_daily, input_days=310, forecast_days=31)
    np.save('./data/processed/X_input.npy', X)
    np.save('./data/processed/y_target.npy', y)
    print("✅ 슬라이딩 윈도우 생성 및 저장 완료\n")
except Exception as e:
    print(f"❌ 슬라이딩 윈도우 실패: {e}")
    exit(1)

# ================================
# 5. 모델 학습
# ================================
try:
    print("🚀 Step 4: Train CNN-LSTM Model")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    split_idx = int(len(X_tensor) * 0.8)
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    model = CNNLSTMModel(input_dim=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 21):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
        val_loss = total_val_loss / len(val_loader.dataset)

        print(f"[{epoch:02d}/20] 🟢 Train Loss: {train_loss:.6f} | 🔵 Val Loss: {val_loss:.6f}")

    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/cnnlstmmodel.pt')
    print("✅ 모델 학습 및 저장 완료\n")

except Exception as e:
    print(f"❌ 모델 학습 실패: {e}")
    exit(1)

# ================================
# 6. 성능 평가
# ================================
try:
    print("📊 Step 5: Evaluate Performance")
    y_val_pred = []
    y_val_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_val_pred.append(pred)
            y_val_true.append(yb.cpu().numpy())

    y_val_pred = np.concatenate(y_val_pred, axis=0)
    y_val_true = np.concatenate(y_val_true, axis=0)

    y_pred_avg = y_val_pred.mean(axis=1)
    y_true_avg = y_val_true.mean(axis=1)

    rmse = mean_squared_error(y_true_avg, y_pred_avg, squared=False)
    mae = mean_absolute_error(y_true_avg, y_pred_avg)
    r2 = r2_score(y_true_avg, y_pred_avg)
    nrmse = rmse / np.std(y_true_avg)

    print("📈 성능 지표 결과:")
    print(f"✅ RMSE   : {rmse:.6f}")
    print(f"✅ MAE    : {mae:.6f}")
    print(f"✅ R²     : {r2:.4f}")
    print(f"✅ NRMSE  : {nrmse:.4f}")

    os.makedirs('./results', exist_ok=True)
    with open('./results/metrics.txt', 'w') as f:
        f.write(f"RMSE   : {rmse:.6f}\n")
        f.write(f"MAE    : {mae:.6f}\n")
        f.write(f"R2     : {r2:.4f}\n")
        f.write(f"NRMSE  : {nrmse:.4f}\n")
    print("📁 성능 지표 저장 완료: ./results/metrics.txt\n")

except Exception as e:
    print(f"❌ 성능 평가 실패: {e}")
    exit(1)

# ================================
# 7. 예측 vs 실제 CPI 그래프 저장
# ================================
try:
    print("📉 Step 6: Plot Prediction vs Actual CPI")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(y_true_avg, label='Actual CPI (Monthly Avg)', marker='o')
    plt.plot(y_pred_avg, label='Predicted CPI (Monthly Avg)', marker='x')
    plt.title('📈 Predicted vs Actual CPI (Validation Set)', fontsize=14)
    plt.xlabel('Validation Sample Index')
    plt.ylabel('Normalized CPI')
    plt.legend()
    plt.grid(True)
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/cpi_prediction_vs_actual.png')
    print("🖼️ 그래프 저장 완료: ./results/cpi_prediction_vs_actual.png\n")

except Exception as e:
    print(f"❌ 예측 그래프 시각화 실패: {e}")
    exit(1)


# ================================
# 완료 메시지
# ================================
print("🎉 전체 파이프라인 성공적으로 완료!")
