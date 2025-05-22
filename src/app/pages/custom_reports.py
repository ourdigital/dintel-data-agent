"""
Custom Reports page rendering logic.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

def render_page(dashboard, start_date: datetime.date, end_date: datetime.date,
                            sources: list[str], metrics: list[str]) -> None:
    """
    Renders the custom reports page.

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
    st.header('사용자 정의 보고서')
    st.write(f"데이터 기간: {start_date} ~ {end_date}")

    # 데이터 로드
    with st.spinner('데이터를 불러오는 중...'):
        df = dashboard.get_data(start_date, end_date, sources)

    if df.empty:
        st.warning('선택한 기간 및 필터에 해당하는 데이터가 없습니다.')
        return

    # 보고서 생성기
    st.subheader('보고서 생성기')

    # 보고서 유형 선택
    report_type = st.selectbox(
        '보고서 유형',
        ['시계열 분석', '비교 분석', '상관관계 분석', '데이터 테이블'],
        key='custom_report_type'
    )

    # 시계열 분석
    if report_type == '시계열 분석':
        # X축 선택
        x_axis_options = ['date']
        if 'date' not in df.columns:
            st.warning("시계열 분석을 위한 'date' 열이 데이터에 없습니다.")
            return
        x_axis = st.selectbox('X축 (날짜)', x_axis_options, key='custom_report_timeseries_x')
        
        # Y축 선택 (복수 선택 가능)
        if not metrics:
            st.warning("분석할 지표가 없습니다. 사이드바에서 지표를 선택해주세요.")
            return
        y_axis = st.multiselect('Y축 (지표)', metrics, default=metrics[0] if metrics else None, key='custom_report_timeseries_y')
        
        # 그룹화 선택
        groupby_options = ['없음', 'source', 'campaign', 'device']
        # Filter out options not present in df columns
        groupby_options = [opt for opt in groupby_options if opt == '없음' or opt in df.columns]
        groupby = st.selectbox('그룹화', groupby_options, key='custom_report_timeseries_group')
        
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
                    y=y_axis[0] if len(y_axis) == 1 else y_axis, # Plotly handles list of y_axis correctly
                    color=groupby,
                    title=f'{groupby}별 시계열 분석',
                    labels={x_axis: '날짜', groupby: groupby}
                )
            
            st.plotly_chart(fig, use_container_width=True)

    # 비교 분석
    elif report_type == '비교 분석':
        # X축 선택
        x_axis_options = ['source', 'campaign', 'device', 'conversion_type']
        x_axis_options = [opt for opt in x_axis_options if opt in df.columns]
        if not x_axis_options:
            st.warning("비교 분석을 위한 범주형 열이 데이터에 없습니다.")
            return
        x_axis = st.selectbox('X축 (범주)', x_axis_options, key='custom_report_comparison_x')
        
        # Y축 선택
        if not metrics:
            st.warning("분석할 지표가 없습니다. 사이드바에서 지표를 선택해주세요.")
            return
        y_axis = st.selectbox('Y축 (지표)', metrics, key='custom_report_comparison_y')
        
        # 정렬 방식
        sort_by = st.radio('정렬 방식', ['값 기준', '범주 기준'], key='custom_report_comparison_sort')
        
        # 그래프 유형
        chart_type = st.radio('그래프 유형', ['막대 그래프', '원형 그래프'], key='custom_report_comparison_chart_type')
        
        # 상위 항목 필터링
        top_n = st.slider('상위 표시 수', min_value=3, max_value=20, value=10, key='custom_report_comparison_top_n')
        
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
        if len(numeric_cols) < 2:
            st.warning("상관관계 분석을 위해서는 최소 2개의 숫자형 열이 필요합니다.")
            return

        # X축 선택
        x_axis = st.selectbox('X축', numeric_cols, key='custom_report_correlation_x')
        
        # Y축 선택
        y_axis_options = [col for col in numeric_cols if col != x_axis]
        if not y_axis_options:
            st.warning("상관관계 분석을 위한 Y축 옵션이 없습니다.")
            return
        y_axis = st.selectbox('Y축', y_axis_options, key='custom_report_correlation_y')
        
        # 그룹화 선택
        groupby_options_corr = ['없음', 'source', 'campaign', 'device']
        groupby_options_corr = [opt for opt in groupby_options_corr if opt == '없음' or opt in df.columns]
        groupby = st.selectbox('색상 구분', groupby_options_corr, key='custom_report_correlation_group')
        
        # 크기 선택
        size_by_options = ['없음'] + [col for col in numeric_cols if col != x_axis and col != y_axis]
        size_by = st.selectbox('크기 변수', size_by_options, key='custom_report_correlation_size')
        
        # 차트 생성
        trendline_ols = 'ols' if st.checkbox('추세선 표시', key='custom_report_correlation_trendline') else None

        if groupby == '없음':
            # 단순 산점도
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                size=size_by if size_by != '없음' else None,
                title=f'{x_axis} vs {y_axis} 상관관계',
                labels={x_axis: x_axis, y_axis: y_axis},
                trendline=trendline_ols
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
                labels={x_axis: x_axis, y_axis: y_axis, groupby: groupby},
                trendline=trendline_ols
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 상관계수 계산
        if x_axis in df.columns and y_axis in df.columns:
            corr = df[[x_axis, y_axis]].corr().iloc[0, 1]
            st.info(f"{x_axis}와 {y_axis} 사이의 상관계수: {corr:.4f}")

    # 데이터 테이블
    elif report_type == '데이터 테이블':
        # 테이블 형식 선택
        table_type = st.radio('테이블 형식', ['요약 테이블', '피벗 테이블'], key='custom_report_table_type')
        
        if table_type == '요약 테이블':
            # 그룹화 선택
            groupby_cols_options = ['source', 'campaign', 'device', 'date', 'conversion_type']
            groupby_cols_options = [opt for opt in groupby_cols_options if opt in df.columns]
            groupby_cols = st.multiselect('그룹화 기준', groupby_cols_options, key='custom_report_summary_table_group')
            
            # 집계 지표 선택
            if not metrics:
                st.warning("분석할 지표가 없습니다. 사이드바에서 지표를 선택해주세요.")
                return
            agg_metrics = st.multiselect('집계 지표', metrics, key='custom_report_summary_table_agg')
            
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
                    file_name="custom_report_summary.csv",
                    mime="text/csv",
                    key='custom_report_summary_download'
                )
        else:  # 피벗 테이블
            # 행 선택
            rows_options = ['source', 'campaign', 'device', 'conversion_type']
            rows_options = [opt for opt in rows_options if opt in df.columns]
            if not rows_options:
                st.warning("피벗 테이블 생성을 위한 행 옵션이 없습니다.")
                return
            rows = st.selectbox('행', rows_options, key='custom_report_pivot_table_rows')
            
            # 열 선택
            columns_options = ['없음', 'source', 'campaign', 'device', 'conversion_type']
            columns_options = [opt for opt in columns_options if opt == '없음' or (opt in df.columns and opt != rows)]
            columns = st.selectbox('열', columns_options, key='custom_report_pivot_table_cols')
            columns = None if columns == '없음' else columns
            
            # 값 선택
            if not metrics:
                st.warning("분석할 지표가 없습니다. 사이드바에서 지표를 선택해주세요.")
                return
            values = st.selectbox('값', metrics, key='custom_report_pivot_table_values')
            
            # 집계 함수 선택
            aggfunc = st.selectbox('집계 함수', ['합계', '평균', '최대값', '최소값'], key='custom_report_pivot_table_aggfunc')
            
            # 집계 함수 매핑
            aggfunc_map = {
                '합계': 'sum',
                '평균': 'mean',
                '최대값': 'max',
                '최소값': 'min'
            }
            
            # 피벗 테이블 생성
            if rows and values:
                try:
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
                        file_name="custom_report_pivot.csv",
                        mime="text/csv",
                        key='custom_report_pivot_download'
                    )
                except Exception as e:
                    st.error(f"피벗 테이블 생성 중 오류 발생: {e}")
    
    # 원시 데이터 표시
    with st.expander("원시 데이터 보기"):
        st.dataframe(df, use_container_width=True)
