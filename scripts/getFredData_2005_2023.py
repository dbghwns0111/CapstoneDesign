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

# 각 리스트 에 대한 시리즈 ID와 이름 매핑
column_rename_map = {
    "KORCPIALLMINMEI": "Total CPI",
    "KORCP010000IXOBM": "Food and non-alcoholic beverages",
    "KORCP020000IXOBM": "Alcoholic beverages, tobacco and narcotics",
    "KORCP030000IXOBM": "Clothing and footwear",
    "KORCP040000IXOBM": "Housing, water, electricity, and fuel",
    "KORCP050000IXOBM": "Household goods and services",
    "KORCP060000IXOBM": "Health",
    "KORCP070000IXOBM": "Transportation",
    "KORCP080000IXOBM": "Communication",
    "KORCP090000IXOBM": "Recreation and culture",
    "KORCP100000IXOBM": "Education",
    "KORCP110000IXOBM": "Restaurants and hotels",
    "KORCP120000IXOBM": "Miscellaneous goods and services",
    "KORCPICORMINMEI": "All items (non-food non-energy)",
    "KORCPIENGMINMEI": "Energy",
    "KORCPGRSE01IXOBM": "Services",
    "KORCPGRLH02IXOBM": "Services less housing",
    "KORCPGRHO02IXOBM": "Housing excluding imputed rentals for housing",
    "KORCP040100IXOBM": "Actual rentals for housing",
    "KORCP040400IXOBM": "Water supply and misc. services relating to dwelling",
    "KORCP040500IXOBM": "Electricity, gas and other fuels",
    "KORCP040300IXOBM": "Maintenance and repair of the dwelling"
}
# 각 시리즈 불러오기 및 이름 변경
for series_id in series_ids:
    series = fred.get_series(series_id)
    series.name = column_rename_map.get(series_id, series_id)  # 이름 변경
    data = pd.concat([data, series], axis=1)

# 인덱스를 datetime 형식으로 변환
data.index = pd.to_datetime(data.index)
data.index.name = 'date'

# 2005년 1월 1일부터 2023년 11월 1일까지 필터링
data = data[(data.index >= '2000-01-01') & (data.index <= '2023-11-01')]

# 결과 확인
print(data)

# CSV 파일로 저장
data.to_csv('data/fred_data_2000_2023.csv')