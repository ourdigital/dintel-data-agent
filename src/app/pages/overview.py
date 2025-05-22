"""
Overview page rendering logic.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def render_page(dashboard, start_date: datetime.date, end_date: datetime.date, 
                       sources: list[str], metrics: list[str]) -> None:
    """
    Renders the overview page.

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
    metrics : List[str]
        A list of selected metrics to display.
    """
    st.header('개요')
    st.write(f"데이터 기간: {start_date} ~ {end_date}")

    # 데이터 로드
    with st.spinner('데이터를 불러오는 중...'):
        df = dashboard.get_data(start_date, end_date, sources)

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
    selected_metric = st.selectbox('지표 선택', metrics, index=0, key='overview_metric_selector')

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
    source_metric = st.selectbox('비교 지표 선택', metrics, index=0, key='overview_source_metric_selector')

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
    campaign_data = campaign_data.sort_values(metrics[0] if metrics else 'impressions', ascending=False) # Handle empty metrics

    # 선택할 지표
    campaign_metric = st.selectbox('캠페인 성과 지표', metrics, index=0, key='overview_campaign_metric_selector')

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
    
    if numeric_cols: # Ensure there are numeric columns to correlate
        # 상관관계 계산
        corr = df[numeric_cols].corr()

        # 히트맵
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('지표 간 상관관계')
        plt.tight_layout()
        
        st.pyplot(fig)
    else:
        st.write("상관관계 분석을 위한 숫자형 데이터가 충분하지 않습니다.")
