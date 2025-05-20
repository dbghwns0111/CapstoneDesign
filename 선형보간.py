import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('data/fred_data.csv')  # 파일명은 사용 환경에 맞게 수정하세요

# 날짜 열을 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['date'])

# 날짜를 인덱스로 설정
df.set_index('date', inplace=True)

# 일별 날짜 인덱스 생성 (시작일 ~ 마지막월의 말일까지)
daily_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max() + pd.offsets.MonthEnd(1) - pd.Timedelta(days=1),
    freq='D'
)

# 월별 데이터를 일별 인덱스로 확장 (중간값은 NaN)
daily_df = df.reindex(daily_index)

# 선형 보간법을 이용해 결측값 채우기
daily_df = daily_df.interpolate(method='linear')

# 인덱스를 다시 'date' 컬럼으로 복원
daily_df.reset_index(inplace=True)
daily_df.rename(columns={'index': 'date'}, inplace=True)

# 결과 저장 (선택 사항)
daily_df.to_csv('data/fred_data_interpolated.csv', index=False)

# 결과 출력 (처음 5행 확인)
print(daily_df.head())
