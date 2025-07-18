"""
Traffic Analysis page rendering logic.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def render_page(dashboard, start_date: datetime.date, end_date: datetime.date, 
                              sources: list[str]) -> None:
    """
    Renders the traffic analysis page.

    Parameters
    ----------
    dashboard : Dashboard
        The main dashboard instance, providing access to data and configurations.
    start_date : datetime.date
        The start date for the data.
    end_date : datetime.date
        The end date for the data.
    sources : List[str]
        A list of selected data sources.
    """
    st.header('트래픽 분석')
    st.write(f"데이터 기간: {start_date} ~ {end_date}")

    # 데이터 로드
    with st.spinner('데이터를 불러오는 중...'):
        df = dashboard.get_data(start_date, end_date, sources)

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
    summary_cols_to_agg = {
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'cost': 'sum'
        }
    # Ensure all expected columns exist before grouping
    existing_cols_for_grouping = [col for col in ['source', 'device'] if col in df.columns]
    
    if not existing_cols_for_grouping:
        st.warning("상세 데이터 테이블을 위한 'source' 또는 'device' 열이 없습니다.")
        return

    summary = df.groupby(existing_cols_for_grouping).agg(summary_cols_to_agg).reset_index()
    
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
