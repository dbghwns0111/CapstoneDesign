import pandas as pd

# 1. 데이터 불러오기 (날짜형으로 변환)
df = pd.read_csv('C:/Users/USER/Desktop/캡스톤디자인/data/품목별_소비자물가지수.csv', parse_dates=['date'])

# 2. 날짜를 인덱스로 설정
df.set_index('date', inplace=True)

# 3. 결측치 처리 (선형 보간 + 마지막 값 채움)
df.interpolate(method='linear', inplace=True)
df.fillna(method='ffill', inplace=True)

# 4. 전월 대비 변화율 (파생 변수 생성)
df_diff = df.pct_change()  # 퍼센트 변화율
df_diff = df_diff.add_suffix('_mom')  # 각 품목별 변화율 컬럼 이름 붙이기

# 5. 원본 품목지수 + 변화율 파생변수 결합
df_final = pd.concat([df, df_diff], axis=1)

# 6. 인덱스를 다시 열로 복원
df_final.reset_index(inplace=True)

# 7. 타겟 생성 (1개월 후 CPI)
# ※ 예: 전체 품목의 평균값 또는 별도 cpi_kr 열이 있다면 그걸 기준으로 생성
df['cpi_kr_target_short'] = df['cpi_kr'].shift(-1)  # 예시 열 이름 '전체지수'

# 7. 인덱스를 열로 복원 후 저장
df.reset_index(inplace=True)
df.to_csv('C:/Users/USER/Desktop/캡스톤디자인/data/품목별_소비자물가지수_전처리.csv', index=False)

print("✅ 전처리 완료 및 저장됨: cpi_items_ready.csv")
