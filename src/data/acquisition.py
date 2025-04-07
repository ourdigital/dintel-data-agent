"""
데이터 수집 모듈.
다양한 데이터 소스에서 데이터를 수집하고 중앙 저장소에 저장합니다.
"""

import os
import logging
import yaml
import pandas as pd
import importlib
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path

# 내부 모듈 가져오기
from src.database.db_manager import DatabaseManager
from src.data.connectors.google_analytics import GoogleAnalyticsConnector

# 다른 커넥터 가져오기 (주석 해제 또는 구현 필요)
# from src.data.connectors.google_ads import GoogleAdsConnector
# from src.data.connectors.meta_ads import MetaAdsConnector
# from src.data.connectors.naver_ads import NaverAdsConnector
# from src.data.connectors.kakao_ads import KakaoAdsConnector

logger = logging.getLogger(__name__)

class DataAcquisition:
    """다양한 소스에서 데이터를 수집하는 클래스."""
    
    def __init__(self, 
                credentials_path: str = "config/api_credentials.yaml", 
                config_path: str = "config/pipeline_config.yaml", 
                db_manager: Optional[DatabaseManager] = None):
        """
        DataAcquisition 초기화.
        
        Parameters
        ----------
        credentials_path : str
            API 인증 정보 파일 경로
        config_path : str
            파이프라인 설정 파일 경로
        db_manager : DatabaseManager, optional
            기존 DB 매니저 객체
        """
        self.credentials_path = credentials_path
        self.config_path = config_path
        self.config = self._load_config()
        
        # DB 매니저 설정
        if db_manager:
            self.db_manager = db_manager
        else:
            self.db_manager = DatabaseManager(config_path)
        
        # 커넥터 초기화
        self.connectors = {}
        self._initialize_connectors()
    
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
            raise
    
    def _initialize_connectors(self) -> None:
        """사용 가능한 모든 데이터 소스 커넥터를 초기화합니다."""
        # 소스 설정 가져오기
        sources_config = self.config['collection']['sources']
        
        # 구현된 커넥터 초기화
        for source_name, source_config in sources_config.items():
            if not source_config.get('enabled', False):
                logger.info(f"소스 '{source_name}'가 비활성화되어 있습니다.")
                continue
                
            try:
                if source_name == 'google_analytics':
                    self.connectors[source_name] = GoogleAnalyticsConnector(
                        credentials_path=self.credentials_path,
                        config_path=self.config_path
                    )
                    logger.info(f"Google Analytics 커넥터 초기화 성공")
                
                # 다른 소스 커넥터 초기화 (주석 해제 또는 구현 필요)
                """
                elif source_name == 'google_ads':
                    self.connectors[source_name] = GoogleAdsConnector(
                        credentials_path=self.credentials_path,
                        config_path=self.config_path
                    )
                    logger.info(f"Google Ads 커넥터 초기화 성공")
                
                elif source_name == 'meta_ads':
                    self.connectors[source_name] = MetaAdsConnector(
                        credentials_path=self.credentials_path,
                        config_path=self.config_path
                    )
                    logger.info(f"Meta Ads 커넥터 초기화 성공")
                
                elif source_name == 'naver_ads':
                    self.connectors[source_name] = NaverAdsConnector(
                        credentials_path=self.credentials_path,
                        config_path=self.config_path
                    )
                    logger.info(f"Naver Ads 커넥터 초기화 성공")
                
                elif source_name == 'kakao_ads':
                    self.connectors[source_name] = KakaoAdsConnector(
                        credentials_path=self.credentials_path,
                        config_path=self.config_path
                    )
                    logger.info(f"Kakao Ads 커넥터 초기화 성공")
                """
                
            except Exception as e:
                logger.error(f"소스 '{source_name}' 커넥터 초기화 실패: {e}")
    
    def collect_data_from_source(self, source_name: str) -> pd.DataFrame:
        """
        지정된 소스에서 데이터를 수집합니다.
        
        Parameters
        ----------
        source_name : str
            데이터 소스 이름
            
        Returns
        -------
        pd.DataFrame
            수집된 데이터가 담긴 DataFrame
        """
        if source_name not in self.connectors:
            logger.warning(f"소스 '{source_name}'에 대한 커넥터가 초기화되지 않았습니다.")
            return pd.DataFrame()
        
        try:
            logger.info(f"'{source_name}'에서 데이터 수집 시작")
            
            connector = self.connectors[source_name]
            
            # 각 커넥터 유형에 맞는 메서드 호출
            if source_name == 'google_analytics':
                data = connector.fetch_data()
                
                # 소스 정보 추가
                if not data.empty:
                    data['source'] = 'Google Analytics'
                
            # 다른 소스 처리 (주석 해제 또는 구현 필요)
            """
            elif source_name == 'google_ads':
                data = connector.fetch_data()
                if not data.empty:
                    data['source'] = 'Google Ads'
                
            elif source_name == 'meta_ads':
                data = connector.fetch_data()
                if not data.empty:
                    data['source'] = 'Meta Ads'
                
            elif source_name == 'naver_ads':
                data = connector.fetch_data()
                if not data.empty:
                    data['source'] = 'Naver Ads'
                
            elif source_name == 'kakao_ads':
                data = connector.fetch_data()
                if not data.empty:
                    data['source'] = 'Kakao Ads'
            """
            
            else:
                logger.warning(f"소스 '{source_name}'에 대한 데이터 수집 메서드가 정의되지 않았습니다.")
                data = pd.DataFrame()
                
            logger.info(f"'{source_name}'에서 데이터 수집 완료, 크기: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"'{source_name}'에서 데이터 수집 실패: {e}")
            return pd.DataFrame()
    
    def collect_data_from_all_sources(self) -> Dict[str, pd.DataFrame]:
        """
        모든 활성화된 소스에서 데이터를 수집합니다.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            소스 이름을 키로, 수집된 데이터를 값으로 하는 딕셔너리
        """
        collected_data = {}
        
        for source_name in self.connectors.keys():
            data = self.collect_data_from_source(source_name)
            
            if not data.empty:
                collected_data[source_name] = data
        
        logger.info(f"모든 소스에서 데이터 수집 완료, 소스 수: {len(collected_data)}")
        return collected_data
    
    def save_to_database(self, data: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> bool:
        """
        DataFrame을 데이터베이스에 저장합니다.
        
        Parameters
        ----------
        data : pd.DataFrame
            저장할 DataFrame
        table_name : str
            테이블 이름
        if_exists : str, default='replace'
            테이블이 존재하는 경우 처리 방법 ('replace', 'append', 'fail')
            
        Returns
        -------
        bool
            성공 여부
        """
        if data.empty:
            logger.warning(f"저장할 데이터가 없습니다: {table_name}")
            return False
        
        try:
            # DB에 저장
            with self.db_manager:
                self.db_manager.dataframe_to_sql(data, table_name, if_exists, index=False)
            
            logger.info(f"데이터를 테이블 '{table_name}'에 저장 성공, 크기: {data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"데이터를 테이블 '{table_name}'에 저장 실패: {e}")
            return False
    
    def save_to_csv(self, data: pd.DataFrame, source_name: str) -> str:
        """
        DataFrame을 CSV 파일로 저장합니다.
        
        Parameters
        ----------
        data : pd.DataFrame
            저장할 DataFrame
        source_name : str
            데이터 소스 이름
            
        Returns
        -------
        str
            저장된 파일 경로
        """
        if data.empty:
            logger.warning(f"저장할 데이터가 없습니다: {source_name}")
            return ""
        
        try:
            # 출력 디렉토리 생성
            output_dir = f"data/raw/{source_name.lower()}/"
            os.makedirs(output_dir, exist_ok=True)
            
            # 현재 날짜를 파일명에 추가
            current_date = datetime.now().strftime("%Y%m%d")
            file_path = os.path.join(output_dir, f"{source_name.lower()}_data_{current_date}.csv")
            
            # CSV로 저장
            data.to_csv(file_path, index=False)
            logger.info(f"'{source_name}' 데이터를 CSV로 저장 완료: {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"'{source_name}' 데이터를 CSV로 저장 실패: {e}")
            return ""
    
    def create_source_tables(self) -> None:
        """
        각 소스에 대한 데이터베이스 테이블을 생성합니다.
        """
        try:
            with self.db_manager:
                # Raw 데이터 테이블
                self.db_manager.create_table_if_not_exists(
                    "raw_data",
                    """
                    CREATE TABLE IF NOT EXISTS raw_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        source TEXT,
                        data_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                
                # 각 소스별 테이블 (필요한 경우)
                self.db_manager.create_table_if_not_exists(
                    "google_analytics_data",
                    """
                    CREATE TABLE IF NOT EXISTS google_analytics_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        sessions INTEGER,
                        pageviews INTEGER,
                        bounce_rate REAL,
                        avg_session_duration REAL,
                        source TEXT DEFAULT 'Google Analytics',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                
                # 처리된 데이터 테이블
                self.db_manager.create_table_if_not_exists(
                    "processed_data",
                    """
                    CREATE TABLE IF NOT EXISTS processed_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        source TEXT,
                        campaign TEXT,
                        impressions INTEGER,
                        clicks INTEGER,
                        conversions INTEGER,
                        cost REAL,
                        ctr REAL,
                        conversion_rate REAL,
                        cost_per_click REAL,
                        cost_per_conversion REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                
            logger.info("데이터베이스 테이블 생성 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 테이블 생성 실패: {e}")
    
    def run_collection_pipeline(self, sources: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        데이터 수집 파이프라인을 실행합니다.
        
        Parameters
        ----------
        sources : List[str], optional
            수집할 소스 목록. None이면 모든 활성화된 소스.
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            소스 이름을 키로, 수집된 데이터를 값으로 하는 딕셔너리
        """
        # 테이블 생성
        self.create_source_tables()
        
        # 소스 목록 확인
        if sources is None:
            # 모든 활성화된 소스에서 데이터 수집
            collected_data = self.collect_data_from_all_sources()
        else:
            # 지정된 소스에서만 데이터 수집
            collected_data = {}
            for source in sources:
                if source in self.connectors:
                    data = self.collect_data_from_source(source)
                    if not data.empty:
                        collected_data[source] = data
                else:
                    logger.warning(f"소스 '{source}'가 초기화되지 않았습니다.")
        
        # 각 소스 데이터 저장
        for source_name, data in collected_data.items():
            # CSV 파일로 저장
            self.save_to_csv(data, source_name)
            
            # 데이터베이스에 저장
            table_name = f"{source_name.lower()}_data"
            self.save_to_database(data, table_name)
        
        logger.info(f"데이터 수집 파이프라인 완료, 처리된 소스 수: {len(collected_data)}")
        return collected_data


# 사용 예제
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 데이터 수집 객체 생성
    acquisition = DataAcquisition()
    
    try:
        # 단일 소스에서 데이터 수집
        source_name = 'google_analytics'
        print(f"\n{source_name}에서 데이터 수집 중...")
        data = acquisition.collect_data_from_source(source_name)
        
        if not data.empty:
            print(f"데이터 수집 성공! 크기: {data.shape}")
            print("\n데이터 미리보기:")
            print(data.head())
            
            # CSV로 저장
            file_path = acquisition.save_to_csv(data, source_name)
            if file_path:
                print(f"\nCSV 파일 저장 위치: {file_path}")
            
            # DB에 저장
            table_name = f"{source_name.lower()}_data"
            success = acquisition.save_to_database(data, table_name)
            if success:
                print(f"\n데이터가 테이블 '{table_name}'에 저장되었습니다.")
            
        else:
            print("데이터 수집 실패 또는 데이터가 없습니다.")
            
        # 전체 파이프라인 실행
        print("\n전체 데이터 수집 파이프라인 실행 중...")
        collected_data = acquisition.run_collection_pipeline()
        
        print(f"\n파이프라인 완료! 수집된 소스 수: {len(collected_data)}")
        for source, df in collected_data.items():
            print(f"- {source}: {df.shape[0]}행 x {df.shape[1]}열")
            
    except Exception as e:
        print(f"오류 발생: {e}")