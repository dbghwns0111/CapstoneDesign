import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
df = pd.read_csv('C:/Users/USER/Desktop/캡스톤디자인/data/cpi_dataset.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# 2. 데이터 기본 확인
print(df.info())
print(df.describe())

# 3. 결측치 확인
print("결측치 수:\n", df.isnull().sum())

# 결측치 처리 예시 (선형 보간 + 마지막 보간)
df.interpolate(method='linear', inplace=True)
df.fillna(method='ffill', inplace=True)

# 4. 시계열 흐름 확인
plt.figure(figsize=(12, 5))
plt.plot(df['cpi_kr'], label='CPI (KR)')
plt.title('CPI 시계열 흐름')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.legend()
plt.grid(True)
plt.show()

# 5. 변수 간 상관관계 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('변수 간 상관관계')
plt.show()

# 6. 예측 대상 생성 (타겟 레이블 시프트)
# 단기: 1개월 후 예측 / 중기: 3개월 후 / 장기: 6개월 후
df['cpi_kr_target_short'] = df['cpi_kr'].shift(-1)
df['cpi_kr_target_mid'] = df['cpi_kr'].shift(-3)
df['cpi_kr_target_long'] = df['cpi_kr'].shift(-6)

# 전처리 완료 데이터 확인
print(df.tail(10))

df.to_csv('C:/Users/USER/Desktop/캡스톤디자인/data/eda_dataset.csv', index=False)