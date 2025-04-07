# 맞춤형 데이터 분석 에이전트 (Customized Data Analysis Agent)

다양한 데이터 소스(웹 분석, 광고 플랫폼, CSV 등)로부터 데이터를 수집, 정제, 분석하고 그 결과를 효과적으로 공유하기 위한 맞춤형 데이터 분석 자동화 솔루션입니다.

## 개요

이 프로젝트는 반복적인 데이터 분석 작업을 자동화하고 분석 효율성을 높이는 데 중점을 둡니다. 다음 4가지 계층으로 구성되어 있습니다:

1. **데이터 수집 계층**: 다양한 소스로부터 데이터를 수집하여 중앙 데이터 저장소에 통합
2. **데이터 준비 계층**: 수집된 데이터를 분석 가능한 형태로 정제 및 준비
3. **데이터 분석 계층**: 준비된 데이터를 활용하여 탐색적 데이터 분석(EDA) 및 심층 분석 수행
4. **결과 보고 계층**: 분석 결과를 시각화하고 이해관계자와 공유하기 위한 리포트 생성

## 지원하는 데이터 소스

- Google Analytics
- Google Search Console
- Meta Ads
- YouTube Analytics
- Google Ads
- Naver/Kakao Ads CSV
- 기타 CSV/Excel 파일

## 설치 방법

```bash
# 저장소 복제
git clone git@github.com:ourdigital/data-analysis-agent.git
cd data-analysis-agent

# 가상환경 생성 및 활성화 (옵션 1: venv 사용)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 가상환경 생성 및 활성화 (옵션 2: conda 사용)
conda create -n data-analysis-agent python=3.12 -y
conda activate data-analysis-agent

# 의존성 패키지 설치
pip install -r requirements.txt

# 개발 모드로 설치
pip install -e .
```

## 설정 방법

1. `config/api_credentials.yaml.example`을 `config/api_credentials.yaml`로 복사하고 필요한 API 인증 정보를 입력합니다.
2. `config/pipeline_config.yaml`에서 데이터 파이프라인 설정을 조정합니다.
3. `.env.example`을 `.env`로 복사하고 데이터베이스 및 API 인증을 위한 환경 변수를 설정합니다.

## 사용 방법

### 데이터 수집 및 처리

```bash
# 모든 소스에서 데이터 수집
python main.py --action collect-all

# 특정 소스에서 데이터 수집
python main.py --action collect --source google_analytics

# 데이터 처리
python main.py --action process
```

### 분석 및 시각화

탐색적 데이터 분석을 위해 Jupyter Notebook을 사용할 수 있습니다:

```bash
jupyter lab notebooks/eda_template.ipynb
```

또는 Streamlit 대시보드를 실행할 수 있습니다:

```bash
streamlit run src/app/dashboard.py
```

## 프로젝트 구조

- `src/data/`: 데이터 수집 및 처리 모듈
- `src/database/`: 데이터베이스 관리 모듈
- `src/analysis/`: 데이터 분석 모듈
- `src/visualization/`: 데이터 시각화 모듈
- `src/app/`: Streamlit 대시보드 앱
- `notebooks/`: Jupyter 노트북
- `tests/`: 단위 테스트

## 기술 스택

- **주요 언어**: Python
- **데이터베이스**: MySQL, SQLite
- **데이터 처리/분석**: Pandas, NumPy, SciPy
- **시각화**: Matplotlib, Seaborn, Plotly
- **분석 환경**: Jupyter Notebook/Lab, Google Colab
- **웹 앱/리포팅**: Streamlit

## 라이선스

MIT

## 기여 방법

1. 이 저장소를 포크합니다
2. 새 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다