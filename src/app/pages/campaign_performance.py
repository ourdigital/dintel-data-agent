"""
Campaign Performance page rendering logic.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

def render_page(dashboard, start_date: datetime.date, end_date: datetime.date,
                                  sources: list[str], metrics: list[str]) -> None:
    """
    Renders the campaign performance analysis page.

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
    st.header('캠페인 성과 분석')
    st.write(f"데이터 기간: {start_date} ~ {end_date}")

    # 데이터 로드
    with st.spinner('데이터를 불러오는 중...'):
        df = dashboard.get_data(start_date, end_date, sources)

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
    campaigns = campaign_summary['campaign'].unique().tolist() # Use unique list of campaigns
    if not campaigns:
        st.warning("분석할 캠페인 데이터가 없습니다.")
        return
    selected_campaign = st.selectbox('캠페인 선택', campaigns, key='campaign_perf_selector')


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
    compare_metric = st.selectbox('비교 지표 선택', metrics, key='campaign_perf_compare_metric')
    if not metrics:
        st.warning("비교할 지표가 선택되지 않았습니다.")
        return
        
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
    source_campaign_agg_metrics = {
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'cost': 'sum'
    }
    # Ensure all expected columns exist before grouping
    source_campaign_group_cols = [col for col in ['source', 'campaign'] if col in df.columns]

    if not source_campaign_group_cols:
        st.warning("소스별 캠페인 성과를 위한 'source' 또는 'campaign' 열이 없습니다.")
        return

    source_campaign = df.groupby(source_campaign_group_cols).agg(source_campaign_agg_metrics).reset_index()
    
    source_campaign['conversion_rate'] = (source_campaign['conversions'] / source_campaign['clicks'] * 100).round(2)

    # 선택된 캠페인 데이터만 필터링
    selected_campaign_data = source_campaign[source_campaign['campaign'] == selected_campaign]

    if selected_campaign_data.empty:
        st.info(f"선택된 캠페인 '{selected_campaign}'에 대한 소스별 데이터가 없습니다.")
    else:
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
