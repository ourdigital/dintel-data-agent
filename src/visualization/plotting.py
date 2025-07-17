"""
데이터 시각화 모듈.
다양한 유형의 시각화를 생성하는 함수들을 제공합니다.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    선택한 DataFrame 열에 대한 상관관계 히트맵을 생성합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 DataFrame
    columns : list of str, optional
        상관관계 행렬에 포함할 열. None인 경우 모든 숫자형 열 사용.
    figsize : tuple of int, default=(10, 8)
        그림 크기
    cmap : str, default='coolwarm'
        히트맵 색상 맵
    annot : bool, default=True
        셀에 상관관계 값 주석 표시 여부
    save_path : str, optional
        그림을 저장할 경로. None이면 그림이 저장되지 않음.
    **kwargs : dict
        seaborn.heatmap에 전달할 추가 인수
        
    Returns
    -------
    fig : plt.Figure
        Figure 객체
    ax : plt.Axes
        Axes 객체
    """
    try:
        # 열 선택
        if columns is None:
            # 숫자형 열만 선택
            numeric_df = df.select_dtypes(include=['number'])
        else:
            numeric_df = df[columns]
        
        # 상관관계 행렬 계산
        corr_matrix = numeric_df.corr()
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 히트맵 생성
        sns.heatmap(
            corr_matrix,
            annot=annot,
            cmap=cmap,
            ax=ax,
            **kwargs
        )
        
        plt.title('상관관계 행렬')
        plt.tight_layout()
        
        # 경로가 제공된 경우 그림 저장
        if save_path:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"상관관계 히트맵이 저장됨: {save_path}")
        
        return fig, ax
        
    except Exception as e:
        logger.error(f"상관관계 히트맵 생성 실패: {e}")
        raise

def plot_feature_importance(
    feature_names: List[str],
    importance_values: List[float],
    title: str = '특성 중요도',
    figsize: Tuple[int, int] = (10, 6),
    color: str = '#1f77b4',
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    특성 중요도에 대한 수평 막대 그래프를 생성합니다.
    
    Parameters
    ----------
    feature_names : list of str
        특성 이름
    importance_values : list of float
        특성에 해당하는 중요도 값
    title : str, default='특성 중요도'
        그래프 제목
    figsize : tuple of int, default=(10, 6)
        그림 크기
    color : str, default='#1f77b4'
        막대 색상
    save_path : str, optional
        그림을 저장할 경로. None이면 그림이 저장되지 않음.
        
    Returns
    -------
    fig : plt.Figure
        Figure 객체
    ax : plt.Axes
        Axes 객체
    """
    try:
        # 중요도 DataFrame 생성
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        # 중요도별 정렬
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 수평 막대 그래프 생성
        ax.barh(importance_df['Feature'], importance_df['Importance'], color=color)
        
        # 제목과 레이블 추가
        ax.set_title(title)
        ax.set_xlabel('중요도')
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 경로가 제공된 경우 그림 저장
        if save_path:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"특성 중요도 그래프가 저장됨: {save_path}")
        
        return fig, ax
        
    except Exception as e:
        logger.error(f"특성 중요도 그래프 생성 실패: {e}")
        raise

def create_time_series_plot(
    df: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    title: str = '시계열 데이터',
    figsize: Tuple[int, int] = (12, 6),
    color_palette: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    시계열 데이터에 대한 선 그래프를 생성합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 DataFrame
    date_column : str
        날짜 열 이름
    value_columns : list of str
        그래프로 표시할 값 열 목록
    title : str, default='시계열 데이터'
        그래프 제목
    figsize : tuple of int, default=(12, 6)
        그림 크기
    color_palette : str, optional
        색상 팔레트
    save_path : str, optional
        그림을 저장할 경로. None이면 그림이 저장되지 않음.
    **kwargs : dict
        pyplot.plot에 전달할 추가 인수
        
    Returns
    -------
    fig : plt.Figure
        Figure 객체
    ax : plt.Axes
        Axes 객체
    """
    try:
        # 날짜 데이터 확인 및 변환
        if df[date_column].dtype != 'datetime64[ns]':
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
        
        # 날짜순으로 정렬
        df = df.sort_values(date_column)
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 색상 팔레트 설정
        if color_palette:
            colors = sns.color_palette(color_palette, len(value_columns))
        else:
            colors = None
        
        # 각 값 열에 대한 선 그래프 생성
        for i, column in enumerate(value_columns):
            color = colors[i] if colors else None
            ax.plot(
                df[date_column], 
                df[column], 
                label=column,
                color=color,
                **kwargs
            )
        
        # 제목과 레이블 설정
        ax.set_title(title)
        ax.set_xlabel('날짜')
        ax.set_ylabel('값')
        
        # x축 레이블 회전
        plt.xticks(rotation=45)
        
        # 범례 표시
        ax.legend()
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 경로가 제공된 경우 그림 저장
        if save_path:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"시계열 그래프가 저장됨: {save_path}")
        
        return fig, ax
        
    except Exception as e:
        logger.error(f"시계열 그래프 생성 실패: {e}")
        raise

def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    title: str = '산점도',
    figsize: Tuple[int, int] = (10, 6),
    add_trendline: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    두 변수 간의 관계를 보여주는 산점도를 생성합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 DataFrame
    x_column : str
        x축 열 이름
    y_column : str
        y축 열 이름
    color_column : str, optional
        점 색상에 사용할 열 이름
    size_column : str, optional
        점 크기에 사용할 열 이름
    title : str, default='산점도'
        그래프 제목
    figsize : tuple of int, default=(10, 6)
        그림 크기
    add_trendline : bool, default=False
        추세선 추가 여부
    save_path : str, optional
        그림을 저장할 경로. None이면 그림이 저장되지 않음.
        
    Returns
    -------
    fig : plt.Figure
        Figure 객체
    ax : plt.Axes
        Axes 객체
    """
    try:
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 크기 열이 있는 경우
        if size_column:
            sizes = df[size_column] / df[size_column].max() * 200 + 20
        else:
            sizes = 60
        
        # 색상 열이 있는 경우
        if color_column:
            scatter = ax.scatter(
                df[x_column],
                df[y_column],
                c=df[color_column],
                s=sizes,
                alpha=0.6,
                cmap='viridis'
            )
            plt.colorbar(scatter, ax=ax, label=color_column)
        else:
            ax.scatter(
                df[x_column],
                df[y_column],
                s=sizes,
                alpha=0.6,
                color='steelblue'
            )
        
        # 추세선 추가
        if add_trendline:
            # 숫자형 데이터만 사용
            mask = pd.notna(df[x_column]) & pd.notna(df[y_column])
            x = df.loc[mask, x_column]
            y = df.loc[mask, y_column]
            
            # 선형 추세선 계산
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # 추세선 그리기
            ax.plot(x, p(x), "r--", alpha=0.8, label=f"추세선: y={z[0]:.4f}x+{z[1]:.4f}")
            ax.legend()
        
        # 제목과 레이블 설정
        ax.set_title(title)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 경로가 제공된 경우 그림 저장
        if save_path:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"산점도가 저장됨: {save_path}")
        
        return fig, ax
        
    except Exception as e:
        logger.error(f"산점도 생성 실패: {e}")
        raise

def create_bar_chart(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str = '막대 그래프',
    figsize: Tuple[int, int] = (10, 6),
    color: Optional[str] = None,
    orientation: str = 'vertical',
    top_n: Optional[int] = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    범주형 데이터에 대한 막대 그래프를 생성합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 DataFrame
    category_column : str
        범주 열 이름
    value_column : str
        값 열 이름
    title : str, default='막대 그래프'
        그래프 제목
    figsize : tuple of int, default=(10, 6)
        그림 크기
    color : str, optional
        막대 색상
    orientation : str, default='vertical'
        그래프 방향 ('vertical' 또는 'horizontal')
    top_n : int, optional
        표시할 상위 항목 수. None이면 모든 항목 표시.
    save_path : str, optional
        그림을 저장할 경로. None이면 그림이 저장되지 않음.
        
    Returns
    -------
    fig : plt.Figure
        Figure 객체
    ax : plt.Axes
        Axes 객체
    """
    try:
        # 데이터 준비
        plot_data = df.groupby(category_column)[value_column].sum().reset_index()
        
        # 값 기준 정렬
        plot_data = plot_data.sort_values(value_column, ascending=False)
        
        # 상위 N개 항목만 표시
        if top_n:
            plot_data = plot_data.head(top_n)
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 막대 그래프 생성
        if orientation == 'vertical':
            if color:
                ax.bar(plot_data[category_column], plot_data[value_column], color=color)
            else:
                ax.bar(plot_data[category_column], plot_data[value_column])
            
            # x축 레이블 회전
            plt.xticks(rotation=45, ha='right')
            
            # 축 레이블 설정
            ax.set_xlabel(category_column)
            ax.set_ylabel(value_column)
        else:  # horizontal
            if color:
                ax.barh(plot_data[category_column], plot_data[value_column], color=color)
            else:
                ax.barh(plot_data[category_column], plot_data[value_column])
            
            # 축 레이블 설정
            ax.set_xlabel(value_column)
            ax.set_ylabel(category_column)
        
        # 제목 설정
        ax.set_title(title)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 경로가 제공된 경우 그림 저장
        if save_path:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"막대 그래프가 저장됨: {save_path}")
        
        return fig, ax
        
    except Exception as e:
        logger.error(f"막대 그래프 생성 실패: {e}")
        raise

def create_pie_chart(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str = '원형 그래프',
    figsize: Tuple[int, int] = (8, 8),
    colors: Optional[List[str]] = None,
    explode: Optional[List[float]] = None,
    show_values: bool = True,
    show_labels: bool = True,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    범주형 데이터에 대한 원형 그래프를 생성합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 DataFrame
    category_column : str
        범주 열 이름
    value_column : str
        값 열 이름
    title : str, default='원형 그래프'
        그래프 제목
    figsize : tuple of int, default=(8, 8)
        그림 크기
    colors : list of str, optional
        각 섹션의 색상 목록
    explode : list of float, optional
        각 섹션이 중심에서 떨어지는 정도
    show_values : bool, default=True
        값 표시 여부
    show_labels : bool, default=True
        레이블 표시 여부
    save_path : str, optional
        그림을 저장할 경로. None이면 그림이 저장되지 않음.
        
    Returns
    -------
    fig : plt.Figure
        Figure 객체
    ax : plt.Axes
        Axes 객체
    """
    try:
        # 데이터 준비
        plot_data = df.groupby(category_column)[value_column].sum().reset_index()
        
        # 값 기준 정렬
        plot_data = plot_data.sort_values(value_column, ascending=False)
        
        # 레이블과 값 추출
        labels = plot_data[category_column] if show_labels else None
        values = plot_data[value_column]
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 원형 그래프 생성
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            explode=explode,
            colors=colors,
            autopct='%1.1f%%' if show_values else None,
            shadow=False,
            startangle=90
        )
        
        # 텍스트 속성 설정
        if show_values:
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
        
        # 제목 설정
        ax.set_title(title)
        
        # 축 비율 균등하게 설정
        ax.axis('equal')
        
        # 경로가 제공된 경우 그림 저장
        if save_path:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"원형 그래프가 저장됨: {save_path}")
        
        return fig, ax
        
    except Exception as e:
        logger.error(f"원형 그래프 생성 실패: {e}")
        raise

def create_interactive_plot(
    df: pd.DataFrame,
    plot_type: str,
    x: Optional[str] = None,
    y: Optional[Union[str, List[str]]] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    title: str = '인터랙티브 그래프',
    **kwargs: Any
) -> go.Figure:
    """
    plotly를 사용하여 인터랙티브 그래프를 생성합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 DataFrame
    plot_type : str
        그래프 유형 ('line', 'bar', 'scatter', 'pie', 'box', 'heatmap')
    x : str, optional
        x축 열 이름
    y : str or list of str, optional
        y축 열 이름 또는 열 이름 목록
    color : str, optional
        색상에 사용할 열 이름
    size : str, optional
        크기에 사용할 열 이름 (산점도에만 적용)
    title : str, default='인터랙티브 그래프'
        그래프 제목
    **kwargs : dict
        plotly 함수에 전달할 추가 인수
        
    Returns
    -------
    fig : Union[go.Figure, go.Figure]
        plotly 그림 객체
    """
    try:
        if plot_type == 'line':
            fig = px.line(
                df, x=x, y=y, color=color,
                title=title,
                **kwargs
            )
        elif plot_type == 'bar':
            fig = px.bar(
                df, x=x, y=y, color=color,
                title=title,
                **kwargs
            )
        elif plot_type == 'scatter':
            fig = px.scatter(
                df, x=x, y=y, color=color, size=size,
                title=title,
                **kwargs
            )
        elif plot_type == 'pie':
            fig = px.pie(
                df, names=x, values=y,
                title=title,
                **kwargs
            )
        elif plot_type == 'box':
            fig = px.box(
                df, x=x, y=y, color=color,
                title=title,
                **kwargs
            )
        elif plot_type == 'heatmap':
            # 히트맵은 피벗 테이블이나 상관관계 행렬이 필요
            if 'z' not in kwargs:
                pivot_data = df.pivot(index=x, columns=color, values=y)
                fig = px.imshow(
                    pivot_data,
                    title=title,
                    **kwargs
                )
            else:
                fig = px.imshow(
                    x=x, y=y, z=kwargs.pop('z'),
                    title=title,
                    **kwargs
                )
        else:
            raise ValueError(f"지원되지 않는 그래프 유형: {plot_type}")
        
        # 레이아웃 업데이트
        fig.update_layout(
            template='plotly_white',
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"인터랙티브 그래프 생성 실패: {e}")
        raise

def save_plot_to_html(
    fig: Union[go.Figure, go.Figure],
    output_path: str,
    include_plotlyjs: bool = True,
    full_html: bool = True
) -> str:
    """
    plotly 그래프를 HTML 파일로 저장합니다.
    
    Parameters
    ----------
    fig : go.Figure or go.Figure
        저장할 plotly 그림 객체
    output_path : str
        출력 파일 경로
    include_plotlyjs : bool, default=True
        plotly.js를 HTML에 포함할지 여부
    full_html : bool, default=True
        완전한 HTML 문서 생성 여부
        
    Returns
    -------
    str
        저장된 파일 경로
    """
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # HTML로 저장
        fig.write_html(
            output_path,
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )
        
        logger.info(f"그래프가 HTML로 저장됨: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"그래프 HTML 저장 실패: {e}")
        raise


# 사용 예제
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=30)
    
    data = {
        'date': dates,
        'impressions': np.random.randint(1000, 5000, len(dates)),
        'clicks': np.random.randint(50, 500, len(dates)),
        'conversions': np.random.randint(1, 50, len(dates)),
        'cost': np.random.uniform(100, 1000, len(dates)),
    }
    
    # CTR 및 기타 파생 지표 계산
    data['ctr'] = [clicks / impressions * 100 for clicks, impressions in zip(data['clicks'], data['impressions'])]
    data['cpc'] = [cost / clicks if clicks > 0 else 0 for cost, clicks in zip(data['cost'], data['clicks'])]
    data['conversion_rate'] = [conversions / clicks * 100 if clicks > 0 else 0 for conversions, clicks in zip(data['conversions'], data['clicks'])]
    
    # 소스 추가
    sources = ['Google Ads', 'Meta Ads', 'Naver Ads']
    data['source'] = np.random.choice(sources, len(dates))
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 출력 디렉토리 생성
    os.makedirs('data/output', exist_ok=True)
    
    # 1. 상관관계 히트맵
    fig1, ax1 = create_correlation_heatmap(
        df,
        columns=['impressions', 'clicks', 'conversions', 'cost', 'ctr', 'cpc', 'conversion_rate'],
        save_path='data/output/correlation_heatmap.png'
    )
    
    # 2. 시계열 그래프
    fig2, ax2 = create_time_series_plot(
        df,
        date_column='date',
        value_columns=['clicks', 'conversions'],
        title='클릭 및 전환 추이',
        save_path='data/output/time_series.png'
    )
    
    # 3. 산점도
    fig3, ax3 = create_scatter_plot(
        df,
        x_column='cost',
        y_column='conversions',
        size_column='clicks',
        color_column='ctr',
        title='비용 vs 전환 (크기: 클릭 수, 색상: CTR)',
        add_trendline=True,
        save_path='data/output/scatter_plot.png'
    )
    
    # 4. 소스별 비교 막대 그래프
    source_data = df.groupby('source')[['impressions', 'clicks', 'conversions', 'cost']].sum().reset_index()
    
    fig4, ax4 = create_bar_chart(
        source_data,
        category_column='source',
        value_column='conversions',
        title='소스별 전환 수',
        save_path='data/output/bar_chart.png'
    )
    
    # 5. 인터랙티브 그래프
    fig5 = create_interactive_plot(
        df,
        plot_type='line',
        x='date',
        y=['clicks', 'conversions'],
        color='source',
        title='소스별 성과 추이'
    )
    
    save_plot_to_html(fig5, 'data/output/interactive_plot.html')
    
    print("모든 그래프가 생성되었습니다. 'data/output' 디렉토리를 확인하세요.")