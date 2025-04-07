"""
Google Analytics 데이터 수집 커넥터.
GoogleAnalytics4 지원 (GA4).
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime, timedelta
import yaml

# Google API 클라이언트 라이브러리 
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GoogleAnalyticsConnector:
    """Google Analytics 데이터를 수집하는 커넥터 클래스."""
    
    def __init__(self, credentials_path: str = "config/api_credentials.yaml", 
                 config_path: str = "config/pipeline_config.yaml"):
        """
        GoogleAnalyticsConnector 초기화.
        
        Parameters
        ----------
        credentials_path : str
            API 인증 정보 파일 경로
        config_path : str
            파이프라인 설정 파일 경로
        """
        self.credentials_path = credentials_path
        self.config_path = config_path
        self.credentials = self._load_credentials()
        self.config = self._load_config()
        self.client = None
        
    def _load_credentials(self) -> Dict[str, Any]:
        """
        API 인증 정보를 로드합니다.
        
        Returns
        -------
        Dict[str, Any]
            인증 정보가 담긴 딕셔너리
        """
        try:
            with open(self.credentials_path, 'r', encoding='utf-8') as f:
                credentials = yaml.safe_load(f)
            return credentials
        except Exception as e:
            logger.error(f"인증 정보 파일 로드 실패: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """
        파이프라인 설정을 로드합니다.
        
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
            raise
    
    def _initialize_client(self) -> None:
        """Google Analytics Data API 클라이언트를 초기화합니다."""
        try:
            # 서비스 계정 사용 방식
            service_account_file = self.credentials['google'].get('service_account_file')
            
            if service_account_file and os.path.exists(service_account_file):
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_file,
                    scopes=["https://www.googleapis.com/auth/analytics.readonly"]
                )
                self.client = BetaAnalyticsDataClient(credentials=credentials)
                logger.info("GA4 API 클라이언트 초기화 성공 (서비스 계정 사용)")
            else:
                # 서비스 계정 파일이 없는 경우 다른 인증 방식을 사용하거나 에러 처리
                logger.warning("서비스 계정 파일이 없거나 유효하지 않음")
                raise ValueError("Google Analytics API 인증에 필요한 서비스 계정 파일이 없습니다.")
        except Exception as e:
            logger.error(f"GA4 API 클라이언트 초기화 실패: {e}")
            raise
    
    def _get_date_range(self) -> tuple:
        """
        설정에서 날짜 범위를 가져옵니다.
        
        Returns
        -------
        tuple
            (시작 날짜, 종료 날짜) 형식의 튜플 (YYYY-MM-DD 문자열)
        """
        date_range = self.config['collection']['default_date_range']
        today = datetime.now().date()
        
        if date_range == "last_7_days":
            start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif date_range == "last_30_days":
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif date_range == "last_90_days":
            start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif date_range == "custom":
            custom_range = self.config['collection']['custom_date_range']
            start_date = custom_range['start_date']
            end_date = custom_range['end_date']
        else:
            # 기본값: 지난 30일
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
        logger.info(f"날짜 범위 설정: {start_date} ~ {end_date}")
        return start_date, end_date
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Google Analytics 데이터를 가져옵니다.
        
        Returns
        -------
        pd.DataFrame
            GA 데이터가 담긴 DataFrame
        """
        if not self.client:
            self._initialize_client()
            
        ga_config = self.config['collection']['sources']['google_analytics']
        
        if not ga_config['enabled']:
            logger.info("Google Analytics 데이터 수집이 비활성화되어 있습니다.")
            return pd.DataFrame()
        
        try:
            property_id = self.credentials['google_analytics']['property_id']
            metrics = ga_config['metrics']
            dimensions = ga_config['dimensions']
            start_date, end_date = self._get_date_range()
            
            # RunReportRequest 객체 생성
            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                metrics=[Metric(name=m) for m in metrics],
                dimensions=[Dimension(name=d) for d in dimensions],
            )
            
            # 보고서 실행
            response = self.client.run_report(request)
            logger.info(f"GA4 데이터 가져오기 성공, 행 수: {len(response.rows)}")
            
            # 응답 처리
            return self._process_response(response, metrics, dimensions)
            
        except Exception as e:
            logger.error(f"GA4 데이터 가져오기 실패: {e}")
            raise
    
    def _process_response(self, response, metrics: List[str], dimensions: List[str]) -> pd.DataFrame:
        """
        GA4 API 응답을 DataFrame으로 변환합니다.
        
        Parameters
        ----------
        response : RunReportResponse
            GA4 API 응답
        metrics : List[str]
            요청한 지표 목록
        dimensions : List[str]
            요청한 차원 목록
            
        Returns
        -------
        pd.DataFrame
            변환된 DataFrame
        """
        rows_data = []
        
        for row in response.rows:
            row_data = {}
            
            # 차원 값 추출
            for i, dimension in enumerate(row.dimension_values):
                row_data[dimensions[i]] = dimension.value
            
            # 지표 값 추출
            for i, metric in enumerate(row.metric_values):
                row_data[metrics[i]] = metric.value
            
            rows_data.append(row_data)
        
        # DataFrame 생성
        df = pd.DataFrame(rows_data)
        
        # 데이터 유형 변환
        for metric in metrics:
            if metric in df:
                try:
                    # 숫자 지표는 float으로 변환
                    df[metric] = pd.to_numeric(df[metric])
                except:
                    # 변환할 수 없는 경우 문자열 유지
                    pass
        
        # 날짜 열이 있으면 datetime으로 변환
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        logger.debug(f"GA4 응답 처리 완료, DataFrame 크기: {df.shape}")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str = "data/raw/google_analytics/") -> str:
        """
        DataFrame을 CSV 파일로 저장합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            저장할 DataFrame
        output_path : str
            출력 디렉토리 경로
            
        Returns
        -------
        str
            저장된 파일 경로
        """
        if df.empty:
            logger.warning("저장할 데이터가 없습니다.")
            return ""
        
        # 디렉토리가 없으면 생성
        os.makedirs(output_path, exist_ok=True)
        
        # 현재 날짜를 파일명에 추가
        current_date = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(output_path, f"ga_data_{current_date}.csv")
        
        # CSV로 저장
        df.to_csv(file_path, index=False)
        logger.info(f"GA 데이터를 CSV로 저장 완료: {file_path}")
        
        return file_path


# 사용 예제
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # GA 커넥터 생성
    ga_connector = GoogleAnalyticsConnector()
    
    try:
        # 데이터 가져오기
        df = ga_connector.fetch_data()
        
        # 결과 출력
        if not df.empty:
            print("\n데이터 미리보기:")
            print(df.head())
            print(f"\n데이터 크기: {df.shape}")
            
            # CSV로 저장
            file_path = ga_connector.save_to_csv(df)
            if file_path:
                print(f"\nCSV 파일 저장 위치: {file_path}")
        else:
            print("데이터가 없거나 GA 데이터 수집이 비활성화되어 있습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")