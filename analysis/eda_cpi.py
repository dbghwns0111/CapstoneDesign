import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 설정
sns.set(style='whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 (윈도우 기준)

# 현재 파일 기준 디렉토리 정의
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '../data/preprocessed_cpi.csv')
output_dir = os.path.join(BASE_DIR, '../results/figures')

# 파일 존재 확인
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ 전처리된 파일을 찾을 수 없습니다: {data_path}")

# 저장 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드
df = pd.read_csv(data_path, parse_dates=['date'])

# 1. CPI 시계열 추이
plt.figure(figsize=(12, 4))
plt.plot(df['date'], df['cpi_kr'], label='CPI')
plt.title('CPI 시계열 추이')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cpi_trend.png'))
plt.show()

# 2. 주요 변수와 CPI 비교 (예시: 실업률)
if 'unemployment_lag1' in df.columns:
    plt.figure(figsize=(12, 4))
    plt.plot(df['date'], df['cpi_kr'], label='CPI')
    plt.plot(df['date'], df['unemployment_lag1'], label='Unemployment (t-1)')
    plt.title('CPI vs Unemployment')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpi_vs_unemployment.png'))
    plt.show()

# 3. 상관관계 히트맵
plt.figure(figsize=(14, 10))
corr = df.drop(columns=['date']).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('상관관계 히트맵')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.show()
