#!/usr/bin/env python
"""
데이터 분석 에이전트 메인 스크립트.
다양한 데이터 소스에서 데이터를 수집, 처리, 분석하고 결과를 보고합니다.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import subprocess

# 내부 모듈 가져오기
from src.utils.logging_config import setup_logging
from src.database.db_manager import DatabaseManager
from src.data.acquisition import DataAcquisition
from src.data.processing import DataProcessor
import src.visualization.plotting as plotting

# 로깅 설정
logger = setup_logging()

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="데이터 분석 에이전트")
    
    parser.add_argument(
        '--action',
        type=str,
        choices=['collect', 'collect-all', 'process', 'analyze', 'visualize', 'dashboard', 'pipeline'],
        default='pipeline',
        help='실행할 작업'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='데이터 소스 (action이 collect인 경우 필요)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='출력 디렉토리'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/pipeline_config.yaml',
        help='설정 파일 경로'
    )
    
    parser.add_argument(
        '--credentials',
        type=str,
        default='config/api_credentials.yaml',
        help='인증 정보 파일 경로'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화'
    )
    
    return parser.parse_args()

def collect_data(args) -> Dict[str, pd.DataFrame]:
    """
    지정된 소스 또는 모든 소스에서 데이터를 수집합니다.
    
    Parameters
    ----------
    args : argparse.Namespace
        명령줄 인수
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        수집된 데이터
    """
    logger.info("데이터 수집 시작")
    
    # 데이터 수집 객체 생성
    acquisition = DataAcquisition(
        credentials_path=args.credentials,
        config_path=args.config
    )
    
    # 단일 소스 또는 모든 소스에서 데이터 수집
    if args.action == 'collect' and args.source:
        logger.info(f"소스 '{args.source}'에서 데이터 수집 중...")
        source_data = acquisition.collect_data_from_source(args.source)
        
        if source_data.empty:
            logger.warning(f"소스 '{args.source}'에서 데이터를 수집할 수 없습니다.")
            return {}
        
        # CSV 및 DB에 저장
        acquisition.save_to_csv(source_data, args.source)
        table_name = f"{args.source.lower()}_data"
        acquisition.save_to_database(source_data, table_name)
        
        return {args.source: source_data}
    
    else:  # 모든 소스
        logger.info("모든 활성화된 소스에서 데이터 수집 중...")
        return acquisition.run_collection_pipeline()

def process_data(args, collected_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    수집된 데이터를 처리합니다.
    
    Parameters
    ----------
    args : argparse.Namespace
        명령줄 인수
    collected_data : Dict[str, pd.DataFrame], optional
        이미 수집된 데이터. None이면 DB에서 가져옵니다.
        
    Returns
    -------
    pd.DataFrame
        처리된 데이터
    """
    logger.info("데이터 처리 시작")
    
    # 데이터 처리 객체 생성
    processor = DataProcessor(config_path=args.config)
    
    # DB 매니저 생성
    db_manager = DatabaseManager(config_path=args.config)
    
    # 수집된 데이터가 없으면 DB에서 가져옴
    if not collected_data:
        logger.info("데이터베이스에서 원시 데이터 가져오기")
        collected_data = {}
        
        with db_manager:
            # 테이블 목록 가져오기
            tables = db_manager.execute_query_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_data'"
            )
            
            # 각 테이블에서 데이터 가져오기
            for table in tables:
                table_name = table[0]
                source_name = table_name.replace('_data', '')
                
                try:
                    data = db_manager.read_sql_table(table_name)
                    if not data.empty:
                        collected_data[source_name] = data
                        logger.info(f"테이블 '{table_name}'에서 {len(data)} 행 로드됨")
                except Exception as e:
                    logger.error(f"테이블 '{table_name}'에서 데이터 로드 실패: {e}")
    
    # 수집된 데이터가 없으면 빈 DataFrame 반환
    if not collected_data:
        logger.warning("처리할 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 모든 DataFrame 병합
    if len(collected_data) > 1:
        merged_data = processor.merge_dataframes(collected_data)
    else:
        source_name, df = next(iter(collected_data.items()))
        merged_data = df.copy()
    
    # 데이터 처리 파이프라인 실행
    processed_data = processor.process_pipeline(merged_data)
    
    # 처리된 데이터 저장
    processor.save_processed_data(processed_data, f"{args.output_dir}/processed/")
    
    # DB에 처리된 데이터 저장
    with db_manager:
        db_manager.dataframe_to_sql(processed_data, "processed_data", if_exists='replace')
    
    logger.info(f"데이터 처리 완료, 처리된 행 수: {len(processed_data)}")
    return processed_data

def analyze_data(args, processed_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    처리된 데이터를 분석합니다.
    
    Parameters
    ----------
    args : argparse.Namespace
        명령줄 인수
    processed_data : pd.DataFrame, optional
        처리된 데이터. None이면 DB에서 가져옵니다.
        
    Returns
    -------
    Dict[str, Any]
        분석 결과
    """
    logger.info("데이터 분석 시작")
    
    # DB 매니저 생성
    db_manager = DatabaseManager(config_path=args.config)
    
    # 처리된 데이터가 없으면 DB에서 가져옴
    if processed_data is None or processed_data.empty:
        logger.info("데이터베이스에서 처리된 데이터 가져오기")
        
        with db_manager:
            try:
                processed_data = db_manager.read_sql_table("processed_data")
                logger.info(f"처리된 데이터 로드됨, 행 수: {len(processed_data)}")
            except Exception as e:
                logger.error(f"처리된 데이터 로드 실패: {e}")
                return {}
    
    # 데이터가 없으면 빈 결과 반환
    if processed_data.empty:
        logger.warning("분석할 데이터가 없습니다.")
        return {}
    
    # 결과 저장용 디렉토리 생성
    os.makedirs(f"{args.output_dir}/analysis", exist_ok=True)
    
    # 분석 결과 딕셔너리
    analysis_results = {}
    
    # 기본 통계량 계산
    try:
        stats = processed_data.describe().to_dict()
        analysis_results['basic_stats'] = stats
        
        # 통계 결과 CSV로 저장
        stats_df = processed_data.describe().reset_index()
        stats_df.to_csv(f"{args.output_dir}/analysis/basic_stats.csv")
        logger.info("기본 통계량 계산 완료")
    except Exception as e:
        logger.error(f"기본 통계량 계산 실패: {e}")
    
    # 소스별 요약
    try:
        if 'source' in processed_data.columns:
            source_summary = processed_data.groupby('source').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'cost': 'sum'
            }).to_dict()
            
            analysis_results['source_summary'] = source_summary
            
            # 소스별 요약 CSV로 저장
            source_summary_df = processed_data.groupby('source').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            source_summary_df.to_csv(f"{args.output_dir}/analysis/source_summary.csv", index=False)
            logger.info("소스별 요약 계산 완료")
    except Exception as e:
        logger.error(f"소스별 요약 계산 실패: {e}")
    
    # 시계열 분석
    try:
        if 'date' in processed_data.columns:
            # 날짜 형식 확인 및 변환
            if processed_data['date'].dtype != 'datetime64[ns]':
                processed_data['date'] = pd.to_datetime(processed_data['date'])
            
            # 일별 집계
            daily_data = processed_data.groupby('date').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            analysis_results['time_series'] = daily_data.to_dict('records')
            
            # 시계열 데이터 CSV로 저장
            daily_data.to_csv(f"{args.output_dir}/analysis/daily_metrics.csv", index=False)
            logger.info("시계열 분석 완료")
    except Exception as e:
        logger.error(f"시계열 분석 실패: {e}")
    
    # 상관관계 분석
    try:
        numeric_cols = processed_data.select_dtypes(include=['number']).columns
        corr_matrix = processed_data[numeric_cols].corr().to_dict()
        
        analysis_results['correlation'] = corr_matrix
        
        # 상관관계 행렬 CSV로 저장
        corr_df = processed_data[numeric_cols].corr().reset_index()
        corr_df.to_csv(f"{args.output_dir}/analysis/correlation_matrix.csv", index=False)
        logger.info("상관관계 분석 완료")
    except Exception as e:
        logger.error(f"상관관계 분석 실패: {e}")
    
    logger.info("데이터 분석 완료")
    return analysis_results

def visualize_data(args, processed_data: Optional[pd.DataFrame] = None, 
                  analysis_results: Optional[Dict[str, Any]] = None) -> bool:
    """
    처리된 데이터와 분석 결과를 시각화합니다.
    
    Parameters
    ----------
    args : argparse.Namespace
        명령줄 인수
    processed_data : pd.DataFrame, optional
        처리된 데이터. None이면 DB에서 가져옵니다.
    analysis_results : Dict[str, Any], optional
        분석 결과. None이면 분석 단계에서 생성된 CSV 파일에서 가져옵니다.
        
    Returns
    -------
    bool
        성공 여부
    """
    logger.info("데이터 시각화 시작")
    
    # DB 매니저 생성
    db_manager = DatabaseManager(config_path=args.config)
    
    # 처리된 데이터가 없으면 DB에서 가져옴
    if processed_data is None or processed_data.empty:
        logger.info("데이터베이스에서 처리된 데이터 가져오기")
        
        with db_manager:
            try:
                processed_data = db_manager.read_sql_table("processed_data")
                logger.info(f"처리된 데이터 로드됨, 행 수: {len(processed_data)}")
            except Exception as e:
                logger.error(f"처리된 데이터 로드 실패: {e}")
                return False
    
    # 데이터가 없으면 실패 반환
    if processed_data.empty:
        logger.warning("시각화할 데이터가 없습니다.")
        return False
    
    # 시각화 결과 저장용 디렉토리 생성
    viz_dir = f"{args.output_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # 날짜 형식 확인 및 변환
        if 'date' in processed_data.columns and processed_data['date'].dtype != 'datetime64[ns]':
            processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        # 1. 상관관계 히트맵
        try:
            numeric_cols = processed_data.select_dtypes(include=['number']).columns
            fig1, ax1 = plotting.create_correlation_heatmap(
                processed_data,
                columns=numeric_cols,
                save_path=f"{viz_dir}/correlation_heatmap.png"
            )
            logger.info("상관관계 히트맵 생성 완료")
        except Exception as e:
            logger.error(f"상관관계 히트맵 생성 실패: {e}")
        
        # 2. 시계열 그래프 (일별 지표)
        if 'date' in processed_data.columns:
            try:
                # 일별 데이터 집계
                daily_data = processed_data.groupby('date').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum',
                    'cost': 'sum'
                }).reset_index()
                
                # 날짜순 정렬
                daily_data = daily_data.sort_values('date')
                
                # 시계열 그래프 생성
                fig2, ax2 = plotting.create_time_series_plot(
                    daily_data,
                    date_column='date',
                    value_columns=['clicks', 'conversions'],
                    title='일별 클릭 및 전환 추이',
                    figsize=(12, 6),
                    save_path=f"{viz_dir}/daily_metrics.png"
                )
                logger.info("시계열 그래프 생성 완료")
                
                # 인터랙티브 시계열 그래프
                fig_interactive = plotting.create_interactive_plot(
                    daily_data,
                    plot_type='line',
                    x='date',
                    y=['impressions', 'clicks', 'conversions', 'cost'],
                    title='일별 지표 추이 (인터랙티브)'
                )
                
                plotting.save_plot_to_html(
                    fig_interactive,
                    f"{viz_dir}/daily_metrics_interactive.html"
                )
                logger.info("인터랙티브 시계열 그래프 생성 완료")
            except Exception as e:
                logger.error(f"시계열 그래프 생성 실패: {e}")
        
        # 3. 소스별 성과 막대 그래프
        if 'source' in processed_data.columns:
            try:
                # 소스별 데이터 집계
                source_data = processed_data.groupby('source').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum',
                    'cost': 'sum'
                }).reset_index()
                
                # 소스별 막대 그래프 생성
                fig3, ax3 = plotting.create_bar_chart(
                    source_data,
                    category_column='source',
                    value_column='conversions',
                    title='소스별 전환 수',
                    figsize=(10, 6),
                    save_path=f"{viz_dir}/source_conversions.png"
                )
                logger.info("소스별 막대 그래프 생성 완료")
                
                # 소스별 비용 막대 그래프
                fig4, ax4 = plotting.create_bar_chart(
                    source_data,
                    category_column='source',
                    value_column='cost',
                    title='소스별 비용',
                    figsize=(10, 6),
                    save_path=f"{viz_dir}/source_cost.png"
                )
                logger.info("소스별 비용 막대 그래프 생성 완료")
            except Exception as e:
                logger.error(f"소스별 그래프 생성 실패: {e}")
        
        # 4. 비용 vs 전환 산점도
        try:
            # 캠페인 또는 소스별 데이터
            scatter_data = processed_data.copy()
            
            if 'campaign' in processed_data.columns:
                group_by = 'campaign'
            elif 'source' in processed_data.columns:
                group_by = 'source'
            else:
                group_by = None
            
            if group_by:
                scatter_data = processed_data.groupby(group_by).agg({
                    'clicks': 'sum',
                    'conversions': 'sum',
                    'cost': 'sum'
                }).reset_index()
                
                # 산점도 생성
                fig5, ax5 = plotting.create_scatter_plot(
                    scatter_data,
                    x_column='cost',
                    y_column='conversions',
                    size_column='clicks',
                    title=f'{group_by}별 비용 vs 전환 (크기: 클릭 수)',
                    add_trendline=True,
                    save_path=f"{viz_dir}/cost_vs_conversions.png"
                )
                logger.info("비용 vs 전환 산점도 생성 완료")
            
        except Exception as e:
            logger.error(f"산점도 생성 실패: {e}")
        
        # 5. 파이 차트 (소스별 또는 캠페인별 비율)
        try:
            if 'source' in processed_data.columns:
                # 소스별 전환 비율
                fig6, ax6 = plotting.create_pie_chart(
                    processed_data,
                    category_column='source',
                    value_column='conversions',
                    title='소스별 전환 비율',
                    save_path=f"{viz_dir}/source_conversion_pie.png"
                )
                logger.info("소스별 전환 비율 파이 차트 생성 완료")
            
            if 'campaign' in processed_data.columns:
                # 캠페인별 비용 비율 (상위 5개)
                campaign_cost = processed_data.groupby('campaign')['cost'].sum().reset_index()
                campaign_cost = campaign_cost.sort_values('cost', ascending=False).head(5)
                
                fig7, ax7 = plotting.create_pie_chart(
                    campaign_cost,
                    category_column='campaign',
                    value_column='cost',
                    title='상위 5개 캠페인별 비용 비율',
                    save_path=f"{viz_dir}/campaign_cost_pie.png"
                )
                logger.info("캠페인별 비용 비율 파이 차트 생성 완료")
        except Exception as e:
            logger.error(f"파이 차트 생성 실패: {e}")
        
        logger.info(f"데이터 시각화 완료. 결과는 '{viz_dir}' 디렉토리에 저장되었습니다.")
        return True
        
    except Exception as e:
        logger.error(f"데이터 시각화 중 오류 발생: {e}")
        return False

def run_dashboard(args):
    """
    Streamlit 대시보드를 실행합니다.
    
    Parameters
    ----------
    args : argparse.Namespace
        명령줄 인수
    """
    logger.info("Streamlit 대시보드 실행 중...")
    
    try:
        dashboard_path = "src/app/dashboard.py"
        
        # Streamlit 실행
        cmd = ["streamlit", "run", dashboard_path, "--server.port=8501"]
        
        if args.debug:
            cmd.append("--logger.level=debug")
        
        logger.info(f"대시보드 명령어 실행: {' '.join(cmd)}")
        
        # Streamlit 프로세스 시작
        process = subprocess.Popen(cmd)
        
        logger.info("대시보드가 시작되었습니다. 브라우저에서 http://localhost:8501로 접속하세요.")
        logger.info("종료하려면 Ctrl+C를 누르세요.")
        
        # 프로세스가 종료될 때까지 대기
        process.wait()
        
    except Exception as e:
        logger.error(f"대시보드 실행 중 오류 발생: {e}")
        return False
    
    return True

def run_pipeline(args):
    """
    전체 데이터 파이프라인을 실행합니다.
    
    Parameters
    ----------
    args : argparse.Namespace
        명령줄 인수
        
    Returns
    -------
    bool
        성공 여부
    """
    logger.info("전체 데이터 파이프라인 실행 시작")
    
    try:
        # 1. 데이터 수집
        collected_data = collect_data(args)
        
        if not collected_data:
            logger.warning("수집된 데이터가 없거나 수집에 실패했습니다.")
            return False
        
        # 2. 데이터 처리
        processed_data = process_data(args, collected_data)
        
        if processed_data.empty:
            logger.warning("처리된 데이터가 없거나 처리에 실패했습니다.")
            return False
        
        # 3. 데이터 분석
        analysis_results = analyze_data(args, processed_data)
        
        # 4. 데이터 시각화
        visualize_data(args, processed_data, analysis_results)
        
        logger.info("전체 데이터 파이프라인 실행 완료")
        
        # 5. 대시보드 실행 여부 확인
        run_dashboard_prompt = input("Streamlit 대시보드를 실행하시겠습니까? (y/n): ").strip().lower()
        
        if run_dashboard_prompt == 'y':
            run_dashboard(args)
        
        return True
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        return False

def main():
    """메인 함수."""
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 디버그 모드 설정
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("디버그 모드가 활성화되었습니다.")
    
    # 선택된 작업 실행
    try:
        if args.action == 'collect' or args.action == 'collect-all':
            collect_data(args)
        elif args.action == 'process':
            process_data(args)
        elif args.action == 'analyze':
            analyze_data(args)
        elif args.action == 'visualize':
            visualize_data(args)
        elif args.action == 'dashboard':
            run_dashboard(args)
        elif args.action == 'pipeline':
            run_pipeline(args)
        else:
            logger.error(f"알 수 없는 작업: {args.action}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램이 중단되었습니다.")
        return 130
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    # 종료 코드 반환
    sys.exit(main())