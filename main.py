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
from src.pipeline import (
    collect_data,
    process_data,
    analyze_data,
    visualize_data,
    run_dashboard,
    run_pipeline
)

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