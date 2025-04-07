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
if os.path.exists('config'):
    ROOT_DIR = '.'
else:
    ROOT_DIR = '../..'

# 내부 모듈 가져오기
import sys
sys.path.append(ROOT_DIR)
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
    
    def __init__(self, config_path: str = f"{ROOT_DIR}/config/pipeline_config.yaml"):
        """
        Dashboard 초기화.
        
        Parameters
        ----------
        config_path : str
            설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.db_manager = DatabaseManager(config_path)
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
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            st.error(f"설정 파일을 로드할 수 없습니다: {e}")
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
        
        # 선택된 페이지 렌더링
        if page == 'overview':
            self.render_overview(start_date, end_date, sources, metrics)
        elif page == 'traffic_analysis':
            self.render_traffic_analysis(start_date, end_date, sources)
        elif page == 'campaign_performance':
            self.render_campaign_performance(start_date, end_date, sources, metrics)
        elif page == 'conversion_analysis':
            self.render_conversion_analysis(start_date, end_date, sources)
        elif page == 'custom_reports':
            self.render_custom_reports(start_date, end_date, sources, metrics)
    
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
    
    def render_overview(self, start_date: datetime.date, end_date: datetime.date, 
                       sources: List[str], metrics: List[str]) -> None:
        """
        개요 페이지를 렌더링합니다.
        
        Parameters
        ----------
        start_date : datetime.date
            시작일
        end_date : datetime.date
            종료일
        sources : List[str]
            데이터 소스 목록
        metrics : List[str]
            표시할 지표 목록
        """
        st.header('개요')
        st.write(f"데이터 기간: {start_date} ~ {end_date}")
        
        # 데이터 로드
        with st.spinner('데이터를 불러오는 중...'):
            df = self.get_data(start_date, end_date, sources)
        
        if df.empty:
            st.warning('선택한 기간 및 필터에 해당하는 데이터가 없습니다.')
            return
        
        # 주요 지표 카드
        st.subheader('주요 지표')
        
        # 전체 합계 계산
        total_impressions = int(df['impressions'].sum())
        total_clicks = int(df['clicks'].sum())
        total_conversions = int(df['conversions'].sum())
        total_cost = float(df['cost'].sum())
        
        # 평균 계산
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        avg_conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        avg_cost_per_click = total_cost / total_clicks if total_clicks > 0 else 0
        avg_cost_per_conversion = total_cost / total_conversions if total_conversions > 0 else 0
        
        # 메트릭 카드 표시
        cols = st.columns(4)
        cols[0].metric("노출 수", f"{total_impressions:,}")
        cols[1].metric("클릭 수", f"{total_clicks:,}")
        cols[2].metric("전환 수", f"{total_conversions:,}")
        cols[3].metric("총 비용", f"₩{total_cost:,.2f}")
        
        cols = st.columns(4)
        cols[0].metric("평균 CTR", f"{avg_ctr:.2f}%")
        cols[1].metric("평균 전환율", f"{avg_conversion_rate:.2f}%")
        cols[2].metric("평균 CPC", f"₩{avg_cost_per_click:.2f}")
        cols[3].metric("평균 CPA", f"₩{avg_cost_per_conversion:.2f}")
        
        # 시계열 차트
        st.subheader('시간에 따른 추이')
        
        # 일별 데이터 집계
        daily_data = df.groupby('date')[metrics].sum().reset_index()
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.sort_values('date')
        
        # 차트 선택기
        selected_metric = st.selectbox('지표 선택', metrics, index=0)
        
        # Plotly를 사용한 인터랙티브 차트
        fig = px.line(
            daily_data, 
            x='date', 
            y=selected_metric,
            title=f'일별 {selected_metric} 추이',
            labels={'date': '날짜', selected_metric: selected_metric}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 소스별 비교
        st.subheader('소스별 비교')
        
        # 소스별 데이터 집계
        source_data = df.groupby('source')[metrics].sum().reset_index()
        
        # 선택할 지표
        source_metric = st.selectbox('비교 지표 선택', metrics, index=0, key='source_metric')
        
        # 바 차트
        fig = px.bar(
            source_data,
            x='source',
            y=source_metric,
            title=f'소스별 {source_metric}',
            labels={'source': '소스', source_metric: source_metric},
            color='source'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 캠페인 성과
        st.subheader('캠페인 성과')
        
        # 캠페인별 데이터 집계
        campaign_data = df.groupby('campaign')[metrics].sum().reset_index()
        campaign_data = campaign_data.sort_values(metrics[0], ascending=False)
        
        # 선택할 지표
        campaign_metric = st.selectbox('캠페인 성과 지표', metrics, index=0, key='campaign_metric')
        
        # 수평 막대 차트
        fig = px.bar(
            campaign_data.head(10),  # 상위 10개만 표시
            y='campaign',
            x=campaign_metric,
            title=f'캠페인별 {campaign_metric} (상위 10개)',
            labels={'campaign': '캠페인', campaign_metric: campaign_metric},
            orientation='h',
            color=campaign_metric
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 상관관계 분석
        st.subheader('지표 간 상관관계')
        
        # 숫자형 열만 선택
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # 상관관계 계산
        corr = df[numeric_cols].corr()
        
        # 히트맵
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('지표 간 상관관계')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    def render_traffic_analysis(self, start_date: datetime.date, end_date: datetime.date, 
                              sources: List[str]) -> None:
        """
        트래픽 분석 페이지를 렌더링합니다.
        
        Parameters
        ----------
        start_date : datetime.date
            시작일
        end_date : datetime.date
            종료일
        sources : List[str]
            데이터 소스 목록
        """
        st.header('트래픽 분석')
        st.write(f"데이터 기간: {start_date} ~ {end_date}")
        
        # 데이터 로드
        with st.spinner('데이터를 불러오는 중...'):
            df = self.get_data(start_date, end_date, sources)
        
        if df.empty:
            st.warning('선택한 기간 및 필터에 해당하는 데이터가 없습니다.')
            return
        
        # 샘플 데이터에 장치 유형 추가 (실제 환경에서는 DB에서 가져옴)
        if 'device' not in df.columns:
            devices = ['Desktop', 'Mobile', 'Tablet']
            df['device'] = np.random.choice(devices, size=len(df))
        
        # 일별 트래픽 추이
        st.subheader('일별 트래픽 추이')
        
        daily_traffic = df.groupby('date')[['impressions', 'clicks']].sum().reset_index()
        daily_traffic['date'] = pd.to_datetime(daily_traffic['date'])
        daily_traffic = daily_traffic.sort_values('date')
        
        # 복합 지표 차트
        fig = go.Figure()
        
        # 노출 데이터 (왼쪽 y축)
        fig.add_trace(
            go.Bar(
                x=daily_traffic['date'],
                y=daily_traffic['impressions'],
                name='노출 수',
                marker_color='lightblue'
            )
        )
        
        # 클릭 데이터 (오른쪽 y축)
        fig.add_trace(
            go.Scatter(
                x=daily_traffic['date'],
                y=daily_traffic['clicks'],
                name='클릭 수',
                marker_color='red',
                yaxis='y2'
            )
        )
        
        # 레이아웃 설정
        fig.update_layout(
            title='일별 노출 및 클릭 추이',
            xaxis=dict(title='날짜'),
            yaxis=dict(title='노출 수', side='left', showgrid=False),
            yaxis2=dict(title='클릭 수', side='right', overlaying='y', showgrid=False),
            legend=dict(x=0.01, y=0.99),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 장치별 트래픽
        st.subheader('장치별 트래픽')
        
        device_traffic = df.groupby('device')[['impressions', 'clicks', 'conversions']].sum().reset_index()
        
        # 탭 생성
        device_tabs = st.tabs(['노출', '클릭', '전환'])
        
        # 노출 탭
        with device_tabs[0]:
            fig = px.pie(
                device_traffic, 
                values='impressions', 
                names='device',
                title='장치별 노출 분포',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 클릭 탭
        with device_tabs[1]:
            fig = px.pie(
                device_traffic, 
                values='clicks', 
                names='device',
                title='장치별 클릭 분포',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 전환 탭
        with device_tabs[2]:
            fig = px.pie(
                device_traffic, 
                values='conversions', 
                names='device',
                title='장치별 전환 분포',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 소스별 트래픽 추이
        st.subheader('소스별 트래픽 추이')
        
        # 소스 및 날짜별 데이터 집계
        source_daily = df.groupby(['source', 'date'])['clicks'].sum().reset_index()
        source_daily['date'] = pd.to_datetime(source_daily['date'])
        source_daily = source_daily.sort_values('date')
        
        # 소스별 선 그래프
        fig = px.line(
            source_daily,
            x='date',
            y='clicks',
            color='source',
            title='소스별 일별 클릭 추이',
            labels={'date': '날짜', 'clicks': '클릭 수', 'source': '소스'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 트래픽 세부 분석
        st.subheader('트래픽 세부 분석')
        
        # 소스 및 디바이스별 CTR 계산
        detailed = df.groupby(['source', 'device']).agg({
            'impressions': 'sum',
            'clicks': 'sum'
        }).reset_index()
        
        detailed['ctr'] = (detailed['clicks'] / detailed['impressions'] * 100).round(2)
        
        # 히트맵 차트
        fig = px.density_heatmap(
            detailed,
            x='source',
            y='device',
            z='ctr',
            title='소스 및 장치별 CTR (%)',
            labels={'source': '소스', 'device': '장치', 'ctr': 'CTR (%)'},
            color_continuous_scale='YlOrRd'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 상세 데이터 테이블
        st.subheader('상세 데이터')
        
        # 집계 데이터 테이블
        summary = df.groupby(['source', 'device']).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'cost': 'sum'
        }).reset_index()
        
        summary['ctr'] = (summary['clicks'] / summary['impressions'] * 100).round(2)
        summary['conversion_rate'] = (summary['conversions'] / summary['clicks'] * 100).round(2)
        summary['cost_per_click'] = (summary['cost'] / summary['clicks']).round(2)
        
        # 데이터 포맷팅
        summary['impressions'] = summary['impressions'].map('{:,.0f}'.format)
        summary['clicks'] = summary['clicks'].map('{:,.0f}'.format)
        summary['conversions'] = summary['conversions'].map('{:,.0f}'.format)
        summary['cost'] = summary['cost'].map('₩{:,.2f}'.format)
        summary['ctr'] = summary['ctr'].map('{:.2f}%'.format)
        summary['conversion_rate'] = summary['conversion_rate'].map('{:.2f}%'.format)
        summary['cost_per_click'] = summary['cost_per_click'].map('₩{:.2f}'.format)
        
        st.dataframe(summary, use_container_width=True)
    
    def render_campaign_performance(self, start_date: datetime.date, end_date: datetime.date,
                                  sources: List[str], metrics: List[str]) -> None:
        """
        캠페인 성과 페이지를 렌더링합니다.
        
        Parameters
        ----------
        start_date : datetime.date
            시작일
        end_date : datetime.date
            종료일
        sources : List[str]
            데이터 소스 목록
        metrics : List[str]
            표시할 지표 목록
        """
        st.header('캠페인 성과 분석')
        st.write(f"데이터 기간: {start_date} ~ {end_date}")
        
        # 데이터 로드
        with st.spinner('데이터를 불러오는 중...'):
            df = self.get_data(start_date, end_date, sources)
        
        if df.empty:
            st.warning('선택한 기간 및 필터에 해당하는 데이터가 없습니다.')
            return
        
        # 캠페인별 성과 요약
        st.subheader('캠페인별 성과 요약')
        
        # 캠페인별 데이터 집계
        campaign_summary = df.groupby('campaign').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'cost': 'sum'
        }).reset_index()
        
        # 파생 지표 계산
        campaign_summary['ctr'] = (campaign_summary['clicks'] / campaign_summary['impressions'] * 100).round(2)
        campaign_summary['conversion_rate'] = (campaign_summary['conversions'] / campaign_summary['clicks'] * 100).round(2)
        campaign_summary['cost_per_click'] = (campaign_summary['cost'] / campaign_summary['clicks']).round(2)
        campaign_summary['cost_per_conversion'] = (campaign_summary['cost'] / campaign_summary['conversions']).round(2)
        campaign_summary['cost_per_conversion'] = campaign_summary['cost_per_conversion'].replace([np.inf, -np.inf], np.nan)
        
        # 캠페인 선택기
        campaigns = campaign_summary['campaign'].tolist()
        selected_campaign = st.selectbox('캠페인 선택', campaigns)
        
        # 선택된 캠페인 데이터
        selected_data = campaign_summary[campaign_summary['campaign'] == selected_campaign].iloc[0]
        
        # 캠페인 성과 카드
        cols = st.columns(4)
        cols[0].metric("노출 수", f"{selected_data['impressions']:,}")
        cols[1].metric("클릭 수", f"{selected_data['clicks']:,}")
        cols[2].metric("전환 수", f"{selected_data['conversions']:,}")
        cols[3].metric("총 비용", f"₩{selected_data['cost']:,.2f}")
        
        cols = st.columns(4)
        cols[0].metric("CTR", f"{selected_data['ctr']:.2f}%")
        cols[1].metric("전환율", f"{selected_data['conversion_rate']:.2f}%")
        cols[2].metric("CPC", f"₩{selected_data['cost_per_click']:.2f}")
        cols[3].metric("CPA", f"₩{selected_data['cost_per_conversion']:.2f}" if not np.isnan(selected_data['cost_per_conversion']) else "N/A")
        
        # 캠페인별 비교
        st.subheader('캠페인 비교')
        
        # 표시할 지표 선택
        compare_metric = st.selectbox('비교 지표 선택', metrics)
        
        # 상위 10개 캠페인만 표시
        top_campaigns = campaign_summary.sort_values(compare_metric, ascending=False).head(10)
        
        # 바 차트
        fig = px.bar(
            top_campaigns,
            y='campaign',
            x=compare_metric,
            title=f'캠페인별 {compare_metric} 비교 (상위 10개)',
            labels={'campaign': '캠페인', compare_metric: compare_metric},
            orientation='h',
            color=compare_metric,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 캠페인 효율성 분석
        st.subheader('캠페인 효율성 분석')
        
        # 산점도 차트
        fig = px.scatter(
            campaign_summary,
            x='cost',
            y='conversions',
            size='clicks',
            color='conversion_rate',
            hover_name='campaign',
            title='비용 vs 전환 수 (버블 크기: 클릭 수, 색상: 전환율)',
            labels={
                'cost': '총 비용', 
                'conversions': '전환 수', 
                'clicks': '클릭 수',
                'conversion_rate': '전환율 (%)'
            },
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 소스별 캠페인 성과
        st.subheader('소스별 캠페인 성과')
        
        # 소스 및 캠페인별 데이터 집계
        source_campaign = df.groupby(['source', 'campaign']).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'cost': 'sum'
        }).reset_index()
        
        source_campaign['conversion_rate'] = (source_campaign['conversions'] / source_campaign['clicks'] * 100).round(2)
        
        # 선택된 캠페인 데이터만 필터링
        selected_campaign_data = source_campaign[source_campaign['campaign'] == selected_campaign]
        
        # 소스별 바 차트
        fig = px.bar(
            selected_campaign_data,
            x='source',
            y=['impressions', 'clicks', 'conversions'],
            title=f'소스별 "{selected_campaign}" 캠페인 성과',
            labels={'source': '소스', 'value': '값', 'variable': '지표'},
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 캠페인 성과 상세 데이터
        st.subheader('캠페인 성과 상세 데이터')
        
        # 캠페인 데이터 포맷팅
        display_data = campaign_summary.copy()
        display_data['impressions'] = display_data['impressions'].map('{:,.0f}'.format)
        display_data['clicks'] = display_data['clicks'].map('{:,.0f}'.format)
        display_data['conversions'] = display_data['conversions'].map('{:,.0f}'.format)
        display_data['cost'] = display_data['cost'].map('₩{:,.2f}'.format)
        display_data['ctr'] = display_data['ctr'].map('{:.2f}%'.format)
        display_data['conversion_rate'] = display_data['conversion_rate'].map('{:.2f}%'.format)
        display_data['cost_per_click'] = display_data['cost_per_click'].map('₩{:.2f}'.format)
        display_data['cost_per_conversion'] = display_data['cost_per_conversion'].apply(
            lambda x: '₩{:.2f}'.format(x) if not np.isnan(x) else 'N/A'
        )
        
        st.dataframe(display_data, use_container_width=True)
    
    def render_conversion_analysis(self, start_date: datetime.date, end_date: datetime.date,
                                sources: List[str]) -> None:
        """
        전환 분석 페이지를 렌더링합니다.
        
        Parameters
        ----------
        start_date : datetime.date
            시작일
        end_date : datetime.date
            종료일
        sources : List[str]
            데이터 소스 목록
        """
        st.header('전환 분석')
        st.write(f"데이터 기간: {start_date} ~ {end_date}")
        
        # 데이터 로드
        with st.spinner('데이터를 불러오는 중...'):
            df = self.get_data(start_date, end_date, sources)
        
        if df.empty:
            st.warning('선택한 기간 및 필터에 해당하는 데이터가 없습니다.')
            return
        
        # 샘플 데이터에 전환 유형 추가 (실제 환경에서는 DB에서 가져옴)
        if 'conversion_type' not in df.columns:
            conversion_types = ['구매', '양식 제출', '회원가입', '다운로드', '문의']
            df['conversion_type'] = np.random.choice(conversion_types, size=len(df))
            
            # 전환 가치 추가
            conversion_values = {
                '구매': (50000, 200000),
                '양식 제출': (10000, 30000),
                '회원가입': (5000, 15000),
                '다운로드': (2000, 10000),
                '문의': (15000, 40000)
            }
            
            df['conversion_value'] = df.apply(
                lambda row: np.random.uniform(*conversion_values[row['conversion_type']]) 
                            if row['conversions'] > 0 else 0, 
                axis=1
            )
        
        # 전환 개요
        st.subheader('전환 개요')
        
        # 전체 전환 지표
        total_conversions = int(df['conversions'].sum())
        total_conversion_value = float(df['conversion_value'].sum()) if 'conversion_value' in df.columns else 0
        avg_conversion_rate = float((df['conversions'].sum() / df['clicks'].sum() * 100)) if df['clicks'].sum() > 0 else 0
        avg_conversion_value = float(total_conversion_value / total_conversions) if total_conversions > 0 else 0
        
        # 전환 지표 카드
        cols = st.columns(4)
        cols[0].metric("총 전환 수", f"{total_conversions:,}")
        cols[1].metric("총 전환 가치", f"₩{total_conversion_value:,.2f}")
        cols[2].metric("평균 전환율", f"{avg_conversion_rate:.2f}%")
        cols[3].metric("평균 전환 가치", f"₩{avg_conversion_value:,.2f}")
        
        # 일별 전환 추이
        st.subheader('일별 전환 추이')
        
        # 일별 데이터 집계
        daily_conversions = df.groupby('date').agg({
            'conversions': 'sum',
            'conversion_value': 'sum' if 'conversion_value' in df.columns else lambda x: 0,
            'cost': 'sum'
        }).reset_index()
        
        daily_conversions['date'] = pd.to_datetime(daily_conversions['date'])
        daily_conversions = daily_conversions.sort_values('date')
        
        # 복합 차트 생성
        fig = go.Figure()
        
        # 전환 수 (막대 차트)
        fig.add_trace(
            go.Bar(
                x=daily_conversions['date'],
                y=daily_conversions['conversions'],
                name='전환 수',
                marker_color='lightgreen'
            )
        )
        
        # 전환 가치 (선 차트, 오른쪽 y축)
        if 'conversion_value' in daily_conversions.columns:
            fig.add_trace(
                go.Scatter(
                    x=daily_conversions['date'],
                    y=daily_conversions['conversion_value'],
                    name='전환 가치',
                    marker_color='darkblue',
                    yaxis='y2'
                )
            )
        
        # 레이아웃 설정
        fig.update_layout(
            title='일별 전환 추이',
            xaxis=dict(title='날짜'),
            yaxis=dict(title='전환 수', side='left', showgrid=False),
            yaxis2=dict(title='전환 가치 (₩)', side='right', overlaying='y', showgrid=False),
            legend=dict(x=0.01, y=0.99),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 전환 유형별 분석
        if 'conversion_type' in df.columns:
            st.subheader('전환 유형별 분석')
            
            # 전환 유형별 데이터 집계
            conversion_type_data = df.groupby('conversion_type').agg({
                'conversions': 'sum',
                'conversion_value': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            # ROAS 계산 (Return On Ad Spend)
            conversion_type_data['roas'] = (conversion_type_data['conversion_value'] / conversion_type_data['cost'] * 100).round(2)
            
            # 도넛 차트 (전환 수)
            fig = px.pie(
                conversion_type_data,
                values='conversions',
                names='conversion_type',
                title='전환 유형별 전환 수 분포',
                hole=0.4
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 전환 가치 차트
            fig = px.bar(
                conversion_type_data.sort_values('conversion_value', ascending=False),
                x='conversion_type',
                y='conversion_value',
                title='전환 유형별 총 전환 가치',
                color='conversion_type',
                labels={'conversion_type': '전환 유형', 'conversion_value': '전환 가치 (₩)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ROAS 차트
            fig = px.bar(
                conversion_type_data.sort_values('roas', ascending=False),
                x='conversion_type',
                y='roas',
                title='전환 유형별 ROAS (투자 수익률)',
                color='roas',
                labels={'conversion_type': '전환 유형', 'roas': 'ROAS (%)'},
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 소스별 전환 성과
        st.subheader('소스별 전환 성과')
        
        # 소스별 데이터 집계
        source_conversion = df.groupby('source').agg({
            'conversions': 'sum',
            'clicks': 'sum',
            'cost': 'sum',
            'conversion_value': 'sum' if 'conversion_value' in df.columns else lambda x: 0
        }).reset_index()
        
        # 파생 지표 계산
        source_conversion['conversion_rate'] = (source_conversion['conversions'] / source_conversion['clicks'] * 100).round(2)
        source_conversion['cost_per_conversion'] = (source_conversion['cost'] / source_conversion['conversions']).round(2)
        source_conversion['roas'] = (source_conversion['conversion_value'] / source_conversion['cost'] * 100).round(2) if 'conversion_value' in df.columns else 0
        
        # 소스별 전환율 차트
        fig = px.bar(
            source_conversion.sort_values('conversion_rate', ascending=False),
            x='source',
            y='conversion_rate',
            title='소스별 전환율',
            color='conversion_rate',
            labels={'source': '소스', 'conversion_rate': '전환율 (%)'},
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 소스별 비용 효율성 차트
        fig = px.scatter(
            source_conversion,
            x='cost_per_conversion',
            y='conversion_rate',
            size='conversions',
            color='source',
            hover_name='source',
            title='소스별 비용 효율성 분석',
            labels={
                'cost_per_conversion': '전환당 비용 (CPA)', 
                'conversion_rate': '전환율 (%)', 
                'conversions': '전환 수'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 전환 성과 요약 표
        st.subheader('전환 성과 요약')
        
        # 데이터 포맷팅
        display_data = source_conversion.copy()
        display_data['conversions'] = display_data['conversions'].map('{:,.0f}'.format)
        display_data['clicks'] = display_data['clicks'].map('{:,.0f}'.format)
        display_data['cost'] = display_data['cost'].map('₩{:,.2f}'.format)
        display_data['conversion_rate'] = display_data['conversion_rate'].map('{:.2f}%'.format)
        display_data['cost_per_conversion'] = display_data['cost_per_conversion'].map('₩{:.2f}'.format)
        
        if 'conversion_value' in display_data.columns:
            display_data['conversion_value'] = display_data['conversion_value'].map('₩{:,.2f}'.format)
            display_data['roas'] = display_data['roas'].map('{:.2f}%'.format)
        
        st.dataframe(display_data, use_container_width=True)
    
    def render_custom_reports(self, start_date: datetime.date, end_date: datetime.date,
                            sources: List[str], metrics: List[str]) -> None:
        """
        사용자 정의 보고서 페이지를 렌더링합니다.
        
        Parameters
        ----------
        start_date : datetime.date
            시작일
        end_date : datetime.date
            종료일
        sources : List[str]
            데이터 소스 목록
        metrics : List[str]
            표시할 지표 목록
        """
        st.header('사용자 정의 보고서')
        st.write(f"데이터 기간: {start_date} ~ {end_date}")
        
        # 데이터 로드
        with st.spinner('데이터를 불러오는 중...'):
            df = self.get_data(start_date, end_date, sources)
        
        if df.empty:
            st.warning('선택한 기간 및 필터에 해당하는 데이터가 없습니다.')
            return
        
        # 보고서 생성기
        st.subheader('보고서 생성기')
        
        # 보고서 유형 선택
        report_type = st.selectbox(
            '보고서 유형',
            ['시계열 분석', '비교 분석', '상관관계 분석', '데이터 테이블']
        )
        
        # 시계열 분석
        if report_type == '시계열 분석':
            # X축 선택
            x_axis = st.selectbox('X축 (날짜)', ['date'])
            
            # Y축 선택 (복수 선택 가능)
            y_axis = st.multiselect('Y축 (지표)', metrics, default=metrics[0] if metrics else None)
            
            # 그룹화 선택
            groupby = st.selectbox('그룹화', ['없음', 'source', 'campaign', 'device'])
            
            if y_axis:
                # 데이터 준비
                if groupby == '없음':
                    # 일별 집계
                    chart_data = df.groupby(x_axis)[y_axis].sum().reset_index()
                    chart_data[x_axis] = pd.to_datetime(chart_data[x_axis])
                    chart_data = chart_data.sort_values(x_axis)
                    
                    # 선 차트
                    fig = px.line(
                        chart_data,
                        x=x_axis,
                        y=y_axis,
                        title='시계열 분석',
                        labels={x_axis: '날짜'}
                    )
                else:
                    # 그룹별 및 일별 집계
                    chart_data = df.groupby([groupby, x_axis])[y_axis].sum().reset_index()
                    chart_data[x_axis] = pd.to_datetime(chart_data[x_axis])
                    chart_data = chart_data.sort_values(x_axis)
                    
                    # 선 차트 (그룹별로 색상 구분)
                    fig = px.line(
                        chart_data,
                        x=x_axis,
                        y=y_axis[0] if len(y_axis) == 1 else y_axis,
                        color=groupby,
                        title=f'{groupby}별 시계열 분석',
                        labels={x_axis: '날짜', groupby: groupby}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 비교 분석
        elif report_type == '비교 분석':
            # X축 선택
            x_axis = st.selectbox('X축 (범주)', ['source', 'campaign', 'device', 'conversion_type'])
            
            # Y축 선택
            y_axis = st.selectbox('Y축 (지표)', metrics)
            
            # 정렬 방식
            sort_by = st.radio('정렬 방식', ['값 기준', '범주 기준'])
            
            # 그래프 유형
            chart_type = st.radio('그래프 유형', ['막대 그래프', '원형 그래프'])
            
            # 상위 항목 필터링
            top_n = st.slider('상위 표시 수', min_value=3, max_value=20, value=10)
            
            # 데이터 준비
            chart_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            
            # 정렬
            if sort_by == '값 기준':
                chart_data = chart_data.sort_values(y_axis, ascending=False)
            else:
                chart_data = chart_data.sort_values(x_axis)
            
            # 상위 항목 필터링
            chart_data = chart_data.head(top_n)
            
            # 차트 생성
            if chart_type == '막대 그래프':
                # 막대 그래프
                fig = px.bar(
                    chart_data,
                    x=x_axis,
                    y=y_axis,
                    title=f'{x_axis}별 {y_axis} 비교',
                    labels={x_axis: x_axis, y_axis: y_axis},
                    color=y_axis
                )
            else:
                # 원형 그래프
                fig = px.pie(
                    chart_data,
                    values=y_axis,
                    names=x_axis,
                    title=f'{x_axis}별 {y_axis} 분포'
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 상관관계 분석
        elif report_type == '상관관계 분석':
            # 숫자형 열만 선택
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # X축 선택
            x_axis = st.selectbox('X축', numeric_cols)
            
            # Y축 선택
            y_axis = st.selectbox('Y축', [col for col in numeric_cols if col != x_axis])
            
            # 그룹화 선택
            groupby = st.selectbox('색상 구분', ['없음', 'source', 'campaign', 'device'])
            
            # 크기 선택
            size_by = st.selectbox('크기 변수', ['없음'] + [col for col in numeric_cols if col != x_axis and col != y_axis])
            
            # 차트 생성
            if groupby == '없음':
                # 단순 산점도
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    size=size_by if size_by != '없음' else None,
                    title=f'{x_axis} vs {y_axis} 상관관계',
                    labels={x_axis: x_axis, y_axis: y_axis},
                    trendline='ols' if st.checkbox('추세선 표시') else None
                )
            else:
                # 그룹별 산점도
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=groupby,
                    size=size_by if size_by != '없음' else None,
                    hover_name=groupby,
                    title=f'{x_axis} vs {y_axis} 상관관계 ({groupby}별)',
                    labels={x_axis: x_axis, y_axis: y_axis, groupby: groupby}
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 상관계수 계산
            corr = df[[x_axis, y_axis]].corr().iloc[0, 1]
            st.info(f"{x_axis}와 {y_axis} 사이의 상관계수: {corr:.4f}")
        
        # 데이터 테이블
        elif report_type == '데이터 테이블':
            # 테이블 형식 선택
            table_type = st.radio('테이블 형식', ['요약 테이블', '피벗 테이블'])
            
            if table_type == '요약 테이블':
                # 그룹화 선택
                groupby_cols = st.multiselect('그룹화 기준', ['source', 'campaign', 'device', 'date', 'conversion_type'])
                
                # 집계 지표 선택
                agg_metrics = st.multiselect('집계 지표', metrics)
                
                if groupby_cols and agg_metrics:
                    # 데이터 집계
                    agg_dict = {metric: 'sum' for metric in agg_metrics}
                    table_data = df.groupby(groupby_cols).agg(agg_dict).reset_index()
                    
                    # 테이블 표시
                    st.dataframe(table_data, use_container_width=True)
                    
                    # CSV 다운로드 버튼
                    csv = table_data.to_csv(index=False)
                    st.download_button(
                        label="CSV로 다운로드",
                        data=csv,
                        file_name="custom_report.csv",
                        mime="text/csv"
                    )
            else:  # 피벗 테이블
                # 행 선택
                rows = st.selectbox('행', ['source', 'campaign', 'device', 'conversion_type'])
                
                # 열 선택
                columns = st.selectbox('열', ['없음', 'source', 'campaign', 'device', 'conversion_type'])
                columns = None if columns == '없음' else columns
                
                # 값 선택
                values = st.selectbox('값', metrics)
                
                # 집계 함수 선택
                aggfunc = st.selectbox('집계 함수', ['합계', '평균', '최대값', '최소값'])
                
                # 집계 함수 매핑
                aggfunc_map = {
                    '합계': 'sum',
                    '평균': 'mean',
                    '최대값': 'max',
                    '최소값': 'min'
                }
                
                # 피벗 테이블 생성
                if rows and values:
                    pivot = pd.pivot_table(
                        df, 
                        values=values, 
                        index=rows, 
                        columns=columns, 
                        aggfunc=aggfunc_map[aggfunc],
                        fill_value=0
                    )
                    
                    # 테이블 표시
                    st.dataframe(pivot, use_container_width=True)
                    
                    # CSV 다운로드 버튼
                    csv = pivot.to_csv()
                    st.download_button(
                        label="CSV로 다운로드",
                        data=csv,
                        file_name="pivot_table.csv",
                        mime="text/csv"
                    )
        
        # 원시 데이터 표시
        with st.expander("원시 데이터 보기"):
            st.dataframe(df, use_container_width=True)


# 애플리케이션 실행
if __name__ == "__main__":
    # 대시보드 생성 및 실행
    dashboard = Dashboard()
    dashboard.run()