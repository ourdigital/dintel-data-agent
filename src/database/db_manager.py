"""
데이터베이스 연결 관리 모듈.
SQLite 및 MySQL 연결을 지원합니다.
"""

import os
import sqlite3
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Union, Dict, Any

# MySQL 연결이 필요한 경우 주석 해제
# import mysql.connector
# from mysql.connector import Error as MySQLError

logger = logging.getLogger(__name__)

class DatabaseManager:
    """데이터베이스 연결 및 쿼리 실행을 관리하는 클래스."""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        DatabaseManager 초기화.
        
        Parameters
        ----------
        config_path : str
            설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.connection = None
        self.db_type = self.config['database']['type']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        설정 파일을 로드합니다.
        
        Parameters
        ----------
        config_path : str
            설정 파일 경로
            
        Returns
        -------
        Dict[str, Any]
            설정 정보가 담긴 딕셔너리
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            raise
    
    def connect(self) -> None:
        """데이터베이스에 연결합니다."""
        try:
            if self.db_type == "sqlite":
                db_path = self.config['database']['sqlite']['db_path']
                # 디렉토리가 없으면 생성
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self.connection = sqlite3.connect(db_path)
                logger.info(f"SQLite 데이터베이스에 연결됨: {db_path}")
            
            elif self.db_type == "mysql":
                # MySQL 사용 시 주석 해제
                """
                mysql_config = self.config['database']['mysql']
                self.connection = mysql.connector.connect(
                    host=mysql_config['host'],
                    port=mysql_config['port'],
                    database=mysql_config['database'],
                    user=mysql_config['user'],
                    password=mysql_config['password']
                )
                logger.info(f"MySQL 데이터베이스에 연결됨: {mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}")
                """
                raise NotImplementedError("MySQL 연결은 아직 구현되지 않았습니다.")
            else:
                raise ValueError(f"지원되지 않는 데이터베이스 유형: {self.db_type}")
                
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            raise
    
    def disconnect(self) -> None:
        """데이터베이스 연결을 종료합니다."""
        if self.connection:
            self.connection.close()
            logger.info("데이터베이스 연결이 종료됨")
            self.connection = None
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> None:
        """
        쿼리를 실행합니다.
        
        Parameters
        ----------
        query : str
            실행할 SQL 쿼리
        params : tuple, optional
            쿼리 파라미터
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            logger.debug(f"쿼리 실행 성공: {query[:50]}...")
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}, 쿼리: {query[:50]}...")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def execute_query_fetchall(self, query: str, params: Optional[tuple] = None) -> list:
        """
        쿼리를 실행하고 모든 결과를 반환합니다.
        
        Parameters
        ----------
        query : str
            실행할 SQL 쿼리
        params : tuple, optional
            쿼리 파라미터
            
        Returns
        -------
        list
            쿼리 결과 리스트
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            logger.debug(f"쿼리 실행 및 결과 조회 성공: {query[:50]}...")
            return result
        except Exception as e:
            logger.error(f"쿼리 실행 또는 결과 조회 실패: {e}, 쿼리: {query[:50]}...")
            raise
        finally:
            cursor.close()
    
    def dataframe_to_sql(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace', index: bool = False) -> None:
        """
        Pandas DataFrame을 SQL 테이블에 저장합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            저장할 DataFrame
        table_name : str
            테이블 이름
        if_exists : str, default='replace'
            테이블이 존재하는 경우 처리 방법 ('replace', 'append', 'fail')
        index : bool, default=False
            DataFrame 인덱스를 테이블에 포함할지 여부
        """
        if not self.connection:
            self.connect()
            
        try:
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=index)
            logger.info(f"DataFrame을 테이블 '{table_name}'에 저장 성공 (크기: {df.shape})")
        except Exception as e:
            logger.error(f"DataFrame을 테이블 '{table_name}'에 저장 실패: {e}")
            raise
    
    def read_sql_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        SQL 쿼리를 실행하고 결과를 DataFrame으로 반환합니다.
        
        Parameters
        ----------
        query : str
            실행할 SQL 쿼리
        params : tuple, optional
            쿼리 파라미터
            
        Returns
        -------
        pd.DataFrame
            쿼리 결과가 담긴 DataFrame
        """
        if not self.connection:
            self.connect()
            
        try:
            if params:
                df = pd.read_sql_query(query, self.connection, params=params)
            else:
                df = pd.read_sql_query(query, self.connection)
            logger.debug(f"SQL 쿼리로 DataFrame 생성 성공: {query[:50]}..., 크기: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"SQL 쿼리로 DataFrame 생성 실패: {e}, 쿼리: {query[:50]}...")
            raise
    
    def read_sql_table(self, table_name: str) -> pd.DataFrame:
        """
        SQL 테이블을 DataFrame으로 읽어옵니다.
        
        Parameters
        ----------
        table_name : str
            테이블 이름
            
        Returns
        -------
        pd.DataFrame
            테이블 데이터가 담긴 DataFrame
        """
        if not self.connection:
            self.connect()
            
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
            logger.debug(f"테이블 '{table_name}'에서 DataFrame 생성 성공, 크기: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"테이블 '{table_name}'에서 DataFrame 생성 실패: {e}")
            raise
    
    def create_table_if_not_exists(self, table_name: str, schema: str) -> None:
        """
        테이블이 존재하지 않으면 생성합니다.
        
        Parameters
        ----------
        table_name : str
            테이블 이름
        schema : str
            테이블 스키마 (CREATE TABLE 구문)
        """
        if not self.connection:
            self.connect()
            
        try:
            self.execute_query(schema)
            logger.info(f"테이블 '{table_name}' 생성 또는 확인 성공")
        except Exception as e:
            logger.error(f"테이블 '{table_name}' 생성 실패: {e}")
            raise
    
    def __enter__(self):
        """컨텍스트 매니저 진입 시 호출."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 시 호출."""
        self.disconnect()
        
    def __del__(self):
        """소멸자."""
        self.disconnect()


# 사용 예제
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 데이터베이스 매니저 생성
    db_manager = DatabaseManager()
    
    # 컨텍스트 매니저로 사용
    with db_manager:
        # 테이블 생성
        db_manager.create_table_if_not_exists(
            "sample_table",
            """
            CREATE TABLE IF NOT EXISTS sample_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value REAL,
                date TEXT
            )
            """
        )
        
        # 데이터 삽입
        db_manager.execute_query(
            "INSERT INTO sample_table (name, value, date) VALUES (?, ?, ?)",
            ("테스트 데이터", 123.45, "2024-04-07")
        )
        
        # 데이터 조회
        results = db_manager.execute_query_fetchall("SELECT * FROM sample_table")
        print("결과:", results)
        
        # DataFrame으로 조회
        df = db_manager.read_sql_table("sample_table")
        print("\nDataFrame으로 조회:")
        print(df)