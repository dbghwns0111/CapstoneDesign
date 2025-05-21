# 필요한 패키지 임포트
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt

# 1. FRED API 키 입력 (https://fredaccount.stlouisfed.org/에서 발급받아야 함)
fred = Fred(api_key='63a56e3b21c9b3d23ea8c9f22fae9ddf')

# 지표 리스트 정의
series_ids = [
    "KORCPIALLMINMEI", "KORCP010000IXOBM", "KORCP020000IXOBM", "KORCP030000IXOBM", "KORCP040000IXOBM",
    "KORCP050000IXOBM", "KORCP060000IXOBM","KORCP070000IXOBM", "KORCP080000IXOBM", "KORCP090000IXOBM",
    "KORCP100000IXOBM", "KORCP110000IXOBM", "KORCP120000IXOBM", "KORCPICORMINMEI", "KORCPIENGMINMEI",
    "KORCPGRSE01IXOBM", "KORCPGRLH02IXOBM", "KORCPGRHO02IXOBM", "KORCP040100IXOBM", "KORCP040400IXOBM",
    "KORCP040500IXOBM", "KORCP040300IXOBM"
]

# 결과 데이터프레임 초기화
data = pd.DataFrame()



# 각 시리즈 불러오기
for series_id in series_ids:
    series = fred.get_series(series_id)
    series.name = series_id
    data = pd.concat([data, series], axis=1)

# 인덱스를 datetime 형식으로 변환
data.index = pd.to_datetime(data.index)
data.index.name = 'date'

# 2005년 1월 1일부터 필터링
data = data[data.index >= pd.Timestamp("2005-01-01")]

# 결과 확인
print(data)

# CSV 파일로 저장
data.to_csv('data/fred_data2.csv')