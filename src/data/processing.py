"""
데이터 정제 및 변환 모듈.
다양한 소스의 데이터를 처리하고 분석을 위한 준비를 수행합니다.
"""

import os
import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

logger = logging.getLogger(__name__)

class DataProcessor:
    """데이터 처리 및 변환을 위한 클래스."""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        DataProcessor 초기화.
        
        Parameters
        ----------
        config_path : str
            설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        
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
    
    def clean_dataframe(self, 
                         df: pd.DataFrame, 
                         drop_duplicates: Optional[bool] = None,
                         handle_missing: Optional[str] = None,
                         drop_na_thresh: Optional[int] = None,
                         na_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        DataFrame을 정제합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            정제할 DataFrame
        drop_duplicates : bool, optional
            중복 행 제거 여부 (설정 파일 값 우선)
        handle_missing : str, optional
            결측치 처리 방법 ('drop', 'fill_mean', 'fill_median', 'fill_zero')
        drop_na_thresh : int, optional
            결측치가 이 값 이상인 행 제거
        na_columns : list of str, optional
            결측치 확인할 열 목록
            
        Returns
        -------
        pd.DataFrame
            정제된 DataFrame
        """
        if df.empty:
            logger.warning("정제할 데이터가 없습니다.")
            return df
        
        # 설정 파일 값 사용
        if drop_duplicates is None:
            drop_duplicates = self.config['processing'].get('clean_duplicates', True)
            
        if handle_missing is None:
            handle_missing = self.config['processing'].get('handle_missing_values', 'drop')
        
        # 원본 데이터 복사
        cleaned_df = df.copy()
        original_shape = cleaned_df.shape
        
        # 중복 제거
        if drop_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
            if original_shape[0] > cleaned_df.shape[0]:
                logger.info(f"중복 행 제거됨: {original_shape[0] - cleaned_df.shape[0]}개")
        
        # 결측치 처리
        if handle_missing == 'drop':
            if na_columns:
                cleaned_df = cleaned_df.dropna(subset=na_columns)
            else:
                if drop_na_thresh:
                    cleaned_df = cleaned_df.dropna(thresh=drop_na_thresh)
                else:
                    cleaned_df = cleaned_df.dropna()
                    
            if original_shape[0] > cleaned_df.shape[0]:
                logger.info(f"결측치 행 제거됨: {original_shape[0] - cleaned_df.shape[0]}개")
                
        elif handle_missing == 'fill_mean':
            # 숫자형 열에만 평균 적용
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            logger.info(f"숫자형 열의 결측치를 평균값으로 대체: {len(numeric_cols)}개 열")
            
        elif handle_missing == 'fill_median':
            # 숫자형 열에만 중앙값 적용
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            logger.info(f"숫자형 열의 결측치를 중앙값으로 대체: {len(numeric_cols)}개 열")
            
        elif handle_missing == 'fill_zero':
            # 모든 열에 0 적용
            cleaned_df = cleaned_df.fillna(0)
            logger.info("모든 결측치를 0으로 대체")
        
        logger.debug(f"데이터 정제 완료, 원본: {original_shape}, 정제 후: {cleaned_df.shape}")
        return cleaned_df
    
    def standardize_date_format(self, df: pd.DataFrame, date_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        날짜 형식을 표준화합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            처리할 DataFrame
        date_cols : list of str, optional
            날짜 열 목록. None이면 자동 감지
            
        Returns
        -------
        pd.DataFrame
            날짜가 표준화된 DataFrame
        """
        if df.empty:
            return df
            
        # 원본 데이터 복사
        processed_df = df.copy()
        
        # 설정에서 날짜 형식 가져오기
        date_format = self.config['processing'].get('standardize_date_format', 'YYYY-MM-DD')
        
        # pandas 형식으로 변환
        if date_format == 'YYYY-MM-DD':
            pd_format = '%Y-%m-%d'
        elif date_format == 'MM/DD/YYYY':
            pd_format = '%m/%d/%Y'
        elif date_format == 'DD/MM/YYYY':
            pd_format = '%d/%m/%Y'
        else:
            pd_format = '%Y-%m-%d'  # 기본값
        
        # 날짜 열 자동 감지
        if date_cols is None:
            # 열 이름에 'date', 'day', 'month', 'year' 등이 포함된 열 찾기
            possible_date_cols = [col for col in processed_df.columns if 
                                 any(key in col.lower() for key in ['date', 'day', 'month', 'year'])]
            
            # 데이터 타입이 object, string, datetime인 열에서만 날짜 변환 시도
            date_cols = []
            for col in possible_date_cols:
                if col in processed_df.columns and processed_df[col].dtype in ['object', 'string', 'datetime64[ns]']:
                    try:
                        # 샘플 값으로 날짜 변환 테스트
                        sample = processed_df[col].dropna().iloc[0] if not processed_df[col].dropna().empty else None
                        if sample and isinstance(sample, str):
                            pd.to_datetime(sample)
                            date_cols.append(col)
                    except:
                        # 변환 실패 시 무시
                        pass
        
        # 날짜 열 처리
        for col in date_cols:
            if col in processed_df.columns:
                try:
                    # datetime으로 변환
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
                    
                    # 지정된 형식으로 다시 문자열 변환
                    processed_df[col] = processed_df[col].dt.strftime(pd_format)
                    
                    logger.info(f"날짜 열 '{col}' 형식 표준화 완료")
                except Exception as e:
                    logger.warning(f"날짜 열 '{col}' 형식 표준화 실패: {e}")
        
        return processed_df
    
    def merge_dataframes(self, 
                         dataframes: Dict[str, pd.DataFrame], 
                         merge_keys: Optional[List[str]] = None,
                         how: str = 'left') -> pd.DataFrame:
        """
        여러 DataFrame을 병합합니다.
        
        Parameters
        ----------
        dataframes : Dict[str, pd.DataFrame]
            소스별 DataFrame 딕셔너리
        merge_keys : list of str, optional
            병합에 사용할 키 목록
        how : str, default='left'
            병합 방식 ('left', 'right', 'inner', 'outer')
            
        Returns
        -------
        pd.DataFrame
            병합된 DataFrame
        """
        if not dataframes:
            logger.warning("병합할 DataFrame이 없습니다.")
            return pd.DataFrame()
        
        # 설정에서 병합 키 가져오기
        if merge_keys is None:
            merge_keys = self.config['processing'].get('merge_keys', ['date'])
        
        # 소스가 1개면 그대로 반환
        if len(dataframes) == 1:
            source_name, df = next(iter(dataframes.items()))
            logger.info(f"DataFrame이 1개뿐이므로 병합 없이 반환: {source_name}, 크기: {df.shape}")
            return df
        
        try:
            # 첫 번째 DataFrame을 기준으로 사용
            sources = list(dataframes.keys())
            result_df = dataframes[sources[0]].copy()
            logger.info(f"병합 기준 DataFrame: {sources[0]}, 크기: {result_df.shape}")
            
            # 나머지 DataFrame 병합
            for i in range(1, len(sources)):
                source = sources[i]
                df = dataframes[source]
                
                # 병합 키가 DataFrame에 있는지 확인
                common_keys = [key for key in merge_keys if key in result_df.columns and key in df.columns]
                
                if not common_keys:
                    logger.warning(f"'{source}' DataFrame과 병합할 공통 키가 없습니다. 병합을 건너뜁니다.")
                    continue
                
                # 소스 구분을 위한 접두사
                result_prefix = f"{sources[0]}_"
                curr_prefix = f"{source}_"
                
                # 중복 열 처리를 위해 접두사 추가
                result_df = pd.merge(
                    result_df, 
                    df,
                    on=common_keys,
                    how=how,
                    suffixes=(f"_{sources[0]}", f"_{source}")
                )
                
                logger.info(f"'{source}' DataFrame 병합 완료, 병합 후 크기: {result_df.shape}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"DataFrame 병합 중 오류 발생: {e}")
            raise
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        열 이름을 표준화합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            처리할 DataFrame
            
        Returns
        -------
        pd.DataFrame
            열 이름이 표준화된 DataFrame
        """
        if df.empty:
            return df
            
        # 원본 데이터 복사
        processed_df = df.copy()
        
        # 열 이름 변환
        rename_dict = {}
        for col in processed_df.columns:
            # 공백을 언더스코어로 변경
            new_col = col.strip().replace(' ', '_')
            
            # 소문자로 변환
            new_col = new_col.lower()
            
            # 특수문자 제거 (언더스코어 제외)
            new_col = ''.join(c for c in new_col if c.isalnum() or c == '_')
            
            # 중복된 언더스코어 제거
            while '__' in new_col:
                new_col = new_col.replace('__', '_')
            
            # 시작/끝 언더스코어 제거
            new_col = new_col.strip('_')
            
            if col != new_col:
                rename_dict[col] = new_col
        
        # 열 이름 변경
        if rename_dict:
            processed_df = processed_df.rename(columns=rename_dict)
            logger.info(f"열 이름 {len(rename_dict)}개 표준화 완료")
        
        return processed_df
    
    def convert_data_types(self, df: pd.DataFrame, type_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        데이터 유형을 변환합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            처리할 DataFrame
        type_map : Dict[str, str], optional
            열 이름과 데이터 유형의 매핑 (예: {'column1': 'int', 'column2': 'float'})
            
        Returns
        -------
        pd.DataFrame
            데이터 유형이 변환된 DataFrame
        """
        if df.empty:
            return df
            
        # 원본 데이터 복사
        processed_df = df.copy()
        
        # 유형 매핑이 없으면 자동 감지
        if type_map is None:
            type_map = {}
            
            # 숫자로 보이는 문자열 열 감지
            for col in processed_df.select_dtypes(include=['object']).columns:
                # 첫 100개 행 샘플링 (또는 전체 행이 100개 미만인 경우)
                sample = processed_df[col].dropna().head(min(100, len(processed_df)))
                
                if sample.empty:
                    continue
                
                # 숫자로 변환 가능한지 확인
                if all(isinstance(x, str) and x.replace('.', '', 1).isdigit() for x in sample if pd.notna(x) and isinstance(x, str)):
                    # 소수점이 있는지 확인
                    if any('.' in str(x) for x in sample if pd.notna(x)):
                        type_map[col] = 'float'
                    else:
                        type_map[col] = 'int'
        
        # 데이터 유형 변환
        for col, dtype in type_map.items():
            if col in processed_df.columns:
                try:
                    if dtype == 'int':
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype(int)
                    elif dtype == 'float':
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    elif dtype == 'str' or dtype == 'string':
                        processed_df[col] = processed_df[col].astype(str)
                    elif dtype == 'bool' or dtype == 'boolean':
                        processed_df[col] = processed_df[col].astype(bool)
                    elif dtype == 'date' or dtype == 'datetime':
                        processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
                    else:
                        processed_df[col] = processed_df[col].astype(dtype)
                        
                    logger.info(f"열 '{col}'의 데이터 유형을 '{dtype}'으로 변환 완료")
                except Exception as e:
                    logger.warning(f"열 '{col}'의 데이터 유형 변환 실패: {e}")
        
        return processed_df
    
    def calculate_derived_metrics(self, df: pd.DataFrame, metrics_config: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        파생 지표를 계산합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            처리할 DataFrame
        metrics_config : Dict[str, Dict], optional
            파생 지표 설정
            
        Returns
        -------
        pd.DataFrame
            파생 지표가 추가된 DataFrame
        """
        if df.empty:
            return df
            
        # 원본 데이터 복사
        processed_df = df.copy()
        
        # 기본 파생 지표 설정
        default_metrics = {
            'ctr': {
                'formula': lambda x: (x['clicks'] / x['impressions']) * 100 if 'clicks' in x and 'impressions' in x else None,
                'required_cols': ['clicks', 'impressions'],
                'description': 'Click-Through Rate (%)'
            },
            'conversion_rate': {
                'formula': lambda x: (x['conversions'] / x['clicks']) * 100 if 'conversions' in x and 'clicks' in x else None,
                'required_cols': ['conversions', 'clicks'],
                'description': 'Conversion Rate (%)'
            },
            'cost_per_click': {
                'formula': lambda x: x['cost'] / x['clicks'] if 'cost' in x and 'clicks' in x and x['clicks'] > 0 else None,
                'required_cols': ['cost', 'clicks'],
                'description': 'Cost per Click'
            },
            'cost_per_conversion': {
                'formula': lambda x: x['cost'] / x['conversions'] if 'cost' in x and 'conversions' in x and x['conversions'] > 0 else None,
                'required_cols': ['cost', 'conversions'],
                'description': 'Cost per Conversion'
            }
        }
        
        # 사용자 정의 지표가 있으면 병합
        if metrics_config:
            default_metrics.update(metrics_config)
        
        # 파생 지표 계산
        for metric_name, config in default_metrics.items():
            # 필요한 열이 모두 있는지 확인
            required_cols = config.get('required_cols', [])
            if all(col in processed_df.columns for col in required_cols):
                try:
                    # 지표 계산
                    formula = config['formula']
                    processed_df[metric_name] = processed_df.apply(formula, axis=1)
                    logger.info(f"파생 지표 '{metric_name}' 계산 완료: {config.get('description', '')}")
                except Exception as e:
                    logger.warning(f"파생 지표 '{metric_name}' 계산 실패: {e}")
        
        return processed_df
    
    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 데이터 처리 파이프라인을 실행합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            처리할 원본 DataFrame
            
        Returns
        -------
        pd.DataFrame
            완전히 처리된 DataFrame
        """
        if df.empty:
            logger.warning("처리할 데이터가 없습니다.")
            return df
            
        logger.info(f"데이터 처리 파이프라인 시작, 원본 데이터 크기: {df.shape}")
        
        # 1. 열 이름 표준화
        processed_df = self.normalize_column_names(df)
        
        # 2. 데이터 정제
        processed_df = self.clean_dataframe(processed_df)
        
        # 3. 데이터 유형 변환
        processed_df = self.convert_data_types(processed_df)
        
        # 4. 날짜 형식 표준화
        processed_df = self.standardize_date_format(processed_df)
        
        # 5. 파생 지표 계산
        processed_df = self.calculate_derived_metrics(processed_df)
        
        logger.info(f"데이터 처리 파이프라인 완료, 처리 후 데이터 크기: {processed_df.shape}")
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = "data/processed/") -> str:
        """
        처리된 DataFrame을 저장합니다.
        
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
        file_path = os.path.join(output_path, f"processed_data_{current_date}.csv")
        
        # CSV로 저장
        df.to_csv(file_path, index=False)
        logger.info(f"처리된 데이터를 CSV로 저장 완료: {file_path}")
        
        return file_path


# 사용 예제
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 데이터 프로세서 생성
    processor = DataProcessor()
    
    # 테스트 데이터 생성
    test_data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Source': ['Google Ads', 'Google Ads', 'Meta Ads', 'Meta Ads', 'Google Ads'],
        'Impressions': [1000, 1200, 800, 950, 1100],
        'Clicks': [50, 60, 40, 45, 55],
        'Cost': [100.5, 120.25, 80.75, 95.5, 110.25],
        'Conversions': [5, 6, 4, 4, 5]
    }
    df = pd.DataFrame(test_data)
    
    try:
        # 데이터 처리 파이프라인 실행
        processed_df = processor.process_pipeline(df)
        
        # 결과 출력
        print("\n원본 데이터:")
        print(df.head())
        
        print("\n처리된 데이터:")
        print(processed_df.head())
        
        # CSV로 저장
        file_path = processor.save_processed_data(processed_df)
        if file_path:
            print(f"\nCSV 파일 저장 위치: {file_path}")
            
    except Exception as e:
        print(f"오류 발생: {e}")