"""
로깅 설정 모듈.
프로젝트 전체에서 사용할 로깅 설정을 제공합니다.
"""

import os
import logging
import yaml
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Any, Optional
from pathlib import Path

def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
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
        # 기본 로깅 사용 (설정 파일 로드 실패 시)
        print(f"경고: 설정 파일 로드 실패: {e}. 기본 로깅 설정을 사용합니다.")
        return {
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': 'logs/pipeline.log',
                'rotate_logs': True,
                'max_log_size_mb': 10
            }
        }

def setup_logging(config_path: str = "config/pipeline_config.yaml", 
                 log_name: Optional[str] = None) -> logging.Logger:
    """
    로깅 시스템을 설정합니다.
    
    Parameters
    ----------
    config_path : str
        설정 파일 경로
    log_name : str, optional
        로거 이름. None이면 루트 로거를 설정합니다.
        
    Returns
    -------
    logging.Logger
        설정된 로거 객체
    """
    # 설정 로드
    config = load_config(config_path)
    log_config = config.get('logging', {})
    
    # 로깅 레벨 가져오기
    level_str = log_config.get('level', 'INFO')
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level_str, logging.INFO)
    
    # 로거 가져오기
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    
    # 핸들러가 이미 있으면 모두 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 기본 포맷터 생성
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # 파일 로깅이 활성화된 경우
    if log_config.get('log_to_file', True):
        log_file = log_config.get('log_file', 'logs/pipeline.log')
        
        # 로그 디렉토리 생성
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # 로그 회전 설정
        if log_config.get('rotate_logs', True):
            # 크기 기반 회전
            max_size_mb = log_config.get('max_log_size_mb', 10)
            max_size_bytes = max_size_mb * 1024 * 1024
            backup_count = log_config.get('backup_count', 3)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            # 일반 파일 핸들러
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    # 프로파게이션 설정 (루트 로거가 아닌 경우)
    if log_name is not None:
        logger.propagate = log_config.get('propagate', False)
    
    return logger

def get_logger(name: str, config_path: str = "config/pipeline_config.yaml") -> logging.Logger:
    """
    지정된 이름으로 로거를 가져옵니다.
    
    Parameters
    ----------
    name : str
        로거 이름
    config_path : str
        설정 파일 경로
        
    Returns
    -------
    logging.Logger
        로거 객체
    """
    return setup_logging(config_path, name)


# 사용 예제
if __name__ == "__main__":
    # 루트 로거 설정
    root_logger = setup_logging()
    root_logger.info("루트 로거가 설정되었습니다.")
    
    # 특정 모듈용 로거 가져오기
    module_logger = get_logger("data_pipeline")
    module_logger.debug("이것은 디버그 메시지입니다.")
    module_logger.info("이것은 정보 메시지입니다.")
    module_logger.warning("이것은 경고 메시지입니다.")
    module_logger.error("이것은 오류 메시지입니다.")
    module_logger.critical("이것은 치명적인 오류 메시지입니다.")
    
    # 다른 모듈용 로거 가져오기
    another_logger = get_logger("data_analysis")
    another_logger.info("다른 모듈의 로거가 설정되었습니다.")
    
    print("로깅 설정이 완료되었습니다. 'logs/pipeline.log' 파일을 확인하세요.")