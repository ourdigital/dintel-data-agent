"""
Conversion Analysis page rendering logic.
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
    Renders the conversion analysis page.

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
    st.header('전환 분석')
    st.write(f"데이터 기간: {start_date} ~ {end_date}")

    # 데이터 로드
    with st.spinner('데이터를 불러오는 중...'):
        df = dashboard.get_data(start_date, end_date, sources)

    if df.empty:
        st.warning('선택한 기간 및 필터에 해당하는 데이터가 없습니다.')
        return

    # 샘플 데이터에 전환 유형 추가 (실제 환경에서는 DB에서 가져옴)
    if 'conversion_type' not in df.columns:
        conversion_types = ['구매', '양식 제출', '회원가입', '다운로드', '문의']
        # Ensure 'conversions' column exists and is numeric before this step
        if 'conversions' not in df.columns or not pd.api.types.is_numeric_dtype(df['conversions']):
             df['conversions'] = np.random.randint(0, 100, size=len(df))


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
    daily_conversions_agg = {
        'conversions': 'sum',
        'cost': 'sum'
    }
    if 'conversion_value' in df.columns:
        daily_conversions_agg['conversion_value'] = 'sum'

    daily_conversions = df.groupby('date').agg(daily_conversions_agg).reset_index()
    
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
        conversion_type_agg = {
            'conversions': 'sum',
            'cost': 'sum'
        }
        if 'conversion_value' in df.columns:
            conversion_type_agg['conversion_value'] = 'sum'
        
        conversion_type_data = df.groupby('conversion_type').agg(conversion_type_agg).reset_index()
        
        if 'conversion_value' in conversion_type_data.columns and 'cost' in conversion_type_data.columns and not conversion_type_data['cost'].eq(0).all():
             conversion_type_data['roas'] = (conversion_type_data['conversion_value'] / conversion_type_data['cost'].replace(0, np.nan) * 100).round(2) # Avoid division by zero
        else:
            conversion_type_data['roas'] = 0.0


        # 도넛 차트 (전환 수)
        fig = px.pie(
            conversion_type_data,
            values='conversions',
            names='conversion_type',
            title='전환 유형별 전환 수 분포',
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)

        if 'conversion_value' in conversion_type_data.columns:
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

        if 'roas' in conversion_type_data.columns:
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
    source_conversion_agg = {
        'conversions': 'sum',
        'clicks': 'sum',
        'cost': 'sum',
    }
    if 'conversion_value' in df.columns:
        source_conversion_agg['conversion_value'] = 'sum'

    source_conversion = df.groupby('source').agg(source_conversion_agg).reset_index()
    
    # 파생 지표 계산
    source_conversion['conversion_rate'] = (source_conversion['conversions'] / source_conversion['clicks'].replace(0, np.nan) * 100).round(2)
    source_conversion['cost_per_conversion'] = (source_conversion['cost'] / source_conversion['conversions'].replace(0, np.nan)).round(2)
    if 'conversion_value' in source_conversion.columns and 'cost' in source_conversion.columns and not source_conversion['cost'].eq(0).all():
        source_conversion['roas'] = (source_conversion['conversion_value'] / source_conversion['cost'].replace(0, np.nan) * 100).round(2)
    else:
        source_conversion['roas'] = 0.0


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
    if 'roas' in display_data.columns:
        display_data['roas'] = display_data['roas'].map('{:.2f}%'.format)
    
    st.dataframe(display_data, use_container_width=True)
