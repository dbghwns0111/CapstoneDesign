import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. CSV 파일 불러오기
df = pd.read_csv('data/fred_data_2000_2023.csv')

# 2. 날짜 제거 및 타겟 변수 설정
df = df.drop(columns=['date'])  # 'date' 열 제거
target_col = 'Total CPI'

# 3. 입력(X), 타겟(y) 분리
X = df.drop(columns=[target_col])
y = df[target_col]

# 4. 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 6. LassoCV 모델 훈련 (5-fold, 반복 10000회)
lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train, y_train)

# 7. 최적 alpha 확인
print(f"🔍 최적 alpha (교차검증 기반): {lasso_cv.alpha_:.6f}")

# 8. 테스트 예측 및 평가
y_pred = lasso_cv.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MSE: {mse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
print(f"🔢 비제로 계수 개수: {(lasso_cv.coef_ != 0).sum()}/{len(lasso_cv.coef_)}")

# 9. 계수 출력 및 정렬
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_cv.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n📊 주요 변수 계수:")
print(coef_df)

# 10. 계수 저장 (CSV)
coef_df.to_csv('data/lassoCV_2000_2023.csv', index=False)

# 11. 시각화: alpha vs CV MSE
mse_path_mean = np.mean(lasso_cv.mse_path_, axis=1)
alphas = lasso_cv.alphas_

plt.figure(figsize=(10, 6))
plt.plot(alphas, mse_path_mean, marker='o', color='blue', label='Mean CV MSE')
plt.axvline(lasso_cv.alpha_, color='red', linestyle='--', label=f'Chosen alpha: {lasso_cv.alpha_:.6f}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('LassoCV: Alpha vs. Mean Cross-Validated MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 알파 값이 0.005로 설정된 경우의 계수 출력
lasso = Lasso(alpha=0.005, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ MSE (alpha=0.005): {mse:.4f}")
print(f"✅ R² Score (alpha=0.005): {r2:.4f}")
print(f"🔢 비제로 계수 개수 (alpha=0.005): {(lasso.coef_ != 0).sum()}/{len(lasso.coef_)}")
# 계수 출력 및 정렬
coef_df_0_005 = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)
print("\n📊 주요 변수 계수 (alpha=0.005):")
print(coef_df_0_005)