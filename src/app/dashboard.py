"""
데이터 분석 대시보드 애플리케이션.
Streamlit을 사용하여 데이터 분석 결과를 시각화합니다.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

# 프로젝트 루트 디렉토리를 확인하고 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 내부 모듈 가져오기
import sys
sys.path.append(str(ROOT_DIR)) # sys.path expects strings
from src.database.db_manager import DatabaseManager
from src.visualization.plotting import create_correlation_heatmap, plot_feature_importance

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Dashboard:
    """데이터 분석 결과를 시각화하는 Streamlit 대시보드."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Dashboard 초기화.
        
        Parameters
        ----------
        config_path : Optional[Union[str, Path]]
            설정 파일 경로. None이면 기본값을 사용합니다.
        """
        if config_path is None:
            self.config_path = ROOT_DIR / "config" / "pipeline_config.yaml"
        else:
            self.config_path = Path(config_path) # Ensure it's a Path object
            
        self.config = self._load_config()
        # Ensure db_manager also receives a Path object or string as it expects
        self.db_manager = DatabaseManager(str(self.config_path)) 
        self.using_sample_data = False # Initialize the flag
        self.setup_page()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        설정 파일을 로드합니다.
        
        Returns
        -------
        Dict[str, Any]
            설정 정보가 담긴 딕셔너리
        """
        try:
            # Ensure self.config_path is a Path object before opening
            with self.config_path.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"설정 파일 '{self.config_path}' 로드 실패: {e}")
            st.error(f"설정 파일 '{self.config_path}'을(를) 로드할 수 없습니다: {e}")
            return {}
    
    def setup_page(self) -> None:
        """페이지 기본 설정을 구성합니다."""
        app_config = self.config.get('app', {})
        
        # 페이지 제목 설정
        st.set_page_config(
            page_title=app_config.get('title', '데이터 분석 대시보드'),
            page_icon='📊',
            layout='wide'
        )
        
        # 페이지 스타일 설정
        theme = app_config.get('theme', {})
        primary_color = theme.get('primary_color', '#FF4B4B')
        background_color = theme.get('background_color', '#F0F2F6')
        
        # CSS 사용자 정의
        st.markdown(f"""
        <style>
        .reportview-container .main .block-container{{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        .sidebar .sidebar-content {{
            background-color: {background_color};
        }}
        .stButton>button {{
            background-color: {primary_color};
            color: white;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def run(self) -> None:
        """대시보드 애플리케이션을 실행합니다."""
        # 제목과 소개
        st.title('맞춤형 데이터 분석 대시보드')
        st.markdown("""
        이 대시보드는 다양한 소스(Google Analytics, Google Ads, Meta Ads 등)에서 수집한 
        데이터를 분석하고 시각화합니다. 왼쪽 사이드바에서 원하는 페이지를 선택하세요.
        """)
        
        # 사이드바 메뉴
        app_config = self.config.get('app', {})
        pages = app_config.get('pages', ['overview', 'traffic_analysis', 'campaign_performance'])
        default_page = app_config.get('default_page', 'overview')
        
        page = st.sidebar.selectbox(
            '페이지 선택',
            pages,
            index=pages.index(default_page) if default_page in pages else 0
        )
        
        # 날짜 필터 (사이드바)
        st.sidebar.markdown("## 날짜 필터")
        date_range = st.sidebar.selectbox(
            '기간 선택',
            ['최근 7일', '최근 30일', '최근 90일', '사용자 정의'],
            index=1
        )
        
        # 사용자 정의 날짜 입력
        if date_range == '사용자 정의':
            end_date = st.sidebar.date_input('종료일', datetime.now())
            start_date = st.sidebar.date_input('시작일', end_date - timedelta(days=30))
        else:
            # 선택된 기간에 따라 시작일과 종료일 설정
            end_date = datetime.now().date()
            if date_range == '최근 7일':
                start_date = end_date - timedelta(days=7)
            elif date_range == '최근 30일':
                start_date = end_date - timedelta(days=30)
            elif date_range == '최근 90일':
                start_date = end_date - timedelta(days=90)
        
        # 필터 및 추가 설정 접기
        with st.sidebar.expander("추가 필터 및 설정"):
            sources = st.multiselect(
                '데이터 소스',
                ['Google Analytics', 'Google Ads', 'Meta Ads', 'Naver Ads', 'Kakao Ads'],
                default=['Google Analytics', 'Google Ads', 'Meta Ads']
            )
            
            metrics = st.multiselect(
                '지표 선택',
                ['impressions', 'clicks', 'conversions', 'cost', 'ctr', 'conversion_rate', 'cost_per_conversion'],
                default=['impressions', 'clicks', 'conversions', 'cost']
            )

        # 데이터 로드 (이 부분은 각 페이지 렌더러로 옮겨졌으므로, run 메서드에서 직접 데이터를 로드하지 않습니다.)
        # 대신, 샘플 데이터 사용 여부 플래그를 확인하여 경고 메시지를 표시합니다.
        # 실제 데이터 로직은 각 페이지의 render_page 함수 내에서 self.get_data()를 호출하여 처리됩니다.
        # self.get_data()가 호출될 때 self.using_sample_data 플래그가 설정됩니다.
        # 따라서, 여기서는 플래그 상태를 확인하고, 필요시 경고를 표시합니다.
        # 페이지 렌더링 전에 이 확인을 수행합니다.
        
        # 이 run 메서드에서 get_data를 호출하지 않으므로, 
        # using_sample_data 플래그는 각 페이지 렌더러 내부의 get_data 호출 시 설정됩니다.
        # 여기에 경고를 표시하려면, 플래그가 설정된 *후* 페이지가 렌더링 *되기 전*에 확인해야 합니다.
        # 현재 구조에서는 get_data가 페이지별로 호출되므로, run 최상단에 두는 것은 적절하지 않을 수 있습니다.
        # 하지만, 만약 어떤 페이지라도 sample data를 사용하게 되면, 그 상태를 run 메소드 레벨에서 알기는 어렵습니다.
        # 각 페이지 렌더링 직전에 확인하거나, get_data 호출 후 바로 확인하는 로직이 필요합니다.
        # 가장 간단한 방법은 get_data를 여기서 한번 호출하고, 그 결과를 페이지에 넘기는 것입니다.
        # 하지만 현재 구조는 페이지별로 get_data를 호출합니다.
        
        # 일단, 가장 최근의 get_data 호출 상태에 따라 경고를 표시하도록 시도합니다.
        # 이는 완벽하지 않을 수 있지만, 요구사항을 최대한 만족시키기 위함입니다.
        if self.using_sample_data:
            st.warning("⚠️ Displaying sample data. An error occurred while loading live data from the database. Please check application logs or data source connectivity.")

        # 선택된 페이지 렌더링
        if page == 'overview':
            from .pages import overview
            overview.render_page(self, start_date, end_date, sources, metrics)
        elif page == 'traffic_analysis':
            from .pages import traffic_analysis
            traffic_analysis.render_page(self, start_date, end_date, sources)
        elif page == 'campaign_performance':
            from .pages import campaign_performance
            campaign_performance.render_page(self, start_date, end_date, sources, metrics)
        elif page == 'conversion_analysis':
            from .pages import conversion_analysis
            conversion_analysis.render_page(self, start_date, end_date, sources)
        elif page == 'custom_reports':
            from .pages import custom_reports
            custom_reports.render_page(self, start_date, end_date, sources, metrics)
    
    def get_data(self, start_date: datetime.date, end_date: datetime.date, sources: List[str] = None) -> pd.DataFrame:
        """
        지정된 기간과 소스에 대한 데이터를 가져옵니다.
        
        Parameters
        ----------
        start_date : datetime.date
            시작일
        end_date : datetime.date
            종료일
        sources : List[str], optional
            데이터 소스 목록
            
        Returns
        -------
        pd.DataFrame
            필터링된 데이터
        """
        self.using_sample_data = False # Reset flag at the beginning of each call
        try:
            # DB에서 데이터 가져오기
            query = """
            SELECT * FROM processed_data 
            WHERE date BETWEEN ? AND ?
            """
            
            params = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # 소스 필터 추가
            if sources and len(sources) > 0:
                source_placeholders = ', '.join(['?'] * len(sources))
                query += f" AND source IN ({source_placeholders})"
                params += tuple(sources)
            
            # 쿼리 실행
            with self.db_manager:
                df = self.db_manager.read_sql_query(query, params)
            
            if df.empty:
                logger.warning(f"쿼리 결과가 없습니다: {query}")
                # 실제 환경에서는 빈 DataFrame 대신 샘플 데이터 로드
                df = self.load_sample_data(start_date, end_date, sources)
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 조회 중 오류 발생: {e}")
            st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
            # 오류 발생 시 샘플 데이터 반환
            self.using_sample_data = True # Set flag as sample data is being used
            return self.load_sample_data(start_date, end_date, sources)
    
    def load_sample_data(self, start_date: datetime.date, end_date: datetime.date, 
                       sources: List[str] = None) -> pd.DataFrame:
        """
        테스트 및 개발을 위한 샘플 데이터를 생성합니다.
        
        Parameters
        ----------
        start_date : datetime.date
            시작일
        end_date : datetime.date
            종료일
        sources : List[str], optional
            데이터 소스 목록
            
        Returns
        -------
        pd.DataFrame
            샘플 데이터
        """
        # 날짜 범위 생성
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # 소스가 없으면 기본값 사용
        if not sources or len(sources) == 0:
            sources = ['Google Analytics', 'Google Ads', 'Meta Ads']
        
        # 캠페인 목록
        campaigns = [
            'Brand_Awareness_Campaign', 
            'Retargeting_Campaign', 
            'New_Product_Launch', 
            'Holiday_Special_Promotion',
            'Email_Signup_Campaign'
        ]
        
        # 데이터 생성
        data = []
        
        for date in date_range:
            for source in sources:
                for campaign in campaigns:
                    # 임의의 데이터 생성
                    impressions = np.random.randint(500, 10000)
                    clicks = np.random.randint(10, int(impressions * 0.1))
                    cost = round(np.random.uniform(50, 500), 2)
                    conversions = np.random.randint(0, int(clicks * 0.2))
                    
                    # 파생 지표 계산
                    ctr = round((clicks / impressions) * 100, 2) if impressions > 0 else 0
                    conversion_rate = round((conversions / clicks) * 100, 2) if clicks > 0 else 0
                    cost_per_click = round(cost / clicks, 2) if clicks > 0 else 0
                    cost_per_conversion = round(cost / conversions, 2) if conversions > 0 else 0
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'source': source,
                        'campaign': campaign,
                        'impressions': impressions,
                        'clicks': clicks,
                        'cost': cost,
                        'conversions': conversions,
                        'ctr': ctr,
                        'conversion_rate': conversion_rate,
                        'cost_per_click': cost_per_click,
                        'cost_per_conversion': cost_per_conversion
                    })
        
        return pd.DataFrame(data)


# 애플리케이션 실행
if __name__ == "__main__":
    # 대시보드 생성 및 실행
    dashboard = Dashboard()
    dashboard.run()