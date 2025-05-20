import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# CSV 파일 불러오기
df = pd.read_csv('data/fred_data_interpolated.csv')

# 날짜 열 제거 및 타겟 변수 설정
df = df.drop(columns=['date'])  # 'date'는 예측에 사용하지 않음
target_col = 'KORCPIALLMINMEI'  # 타겟 변수 이름 (수정 가능)

# 입력 변수(X)와 타겟 변수(y) 분리
X = df.drop(columns=[target_col])
y = df[target_col]

# Min-Max 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Lasso 회귀 모델 학습 (alpha 값은 정규화 강도)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 예측
y_pred = lasso.predict(X_test)

# 성능 평가 지표 출력
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ MSE: {mse:.4f}")
print(f"✅ R^2 Score: {r2:.4f}")

# 중요 변수 확인
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
})
important_features = coef_df[coef_df['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False)

print("\n🔍 중요 변수 (Lasso가 선택한 변수):")
print(important_features)
