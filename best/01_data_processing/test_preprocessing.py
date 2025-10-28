#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 데이터 전처리 시스템 테스트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from configs.best_config import CONFIG, DatasetType, logger
from optimized_preprocessing import OptimizedDataProcessor

def test_single_dataset():
    """단일 데이터셋 테스트 (작은 규모)"""
    logger.info("🧪 단일 데이터셋 테스트 시작")
    
    # 가장 작은 데이터셋으로 테스트
    test_dataset = DatasetType.GREENHOUSE_SINGLE
    
    processor = OptimizedDataProcessor(
        output_dir="D:/Nong-View/best/results/test_output",
        enable_quality_filter=True
    )
    
    try:
        result = processor.process_single_dataset(test_dataset)
        
        logger.info("✅ 테스트 성공!")
        logger.info(f"처리 결과: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        return False

def test_config_validation():
    """설정 검증 테스트"""
    logger.info("🔧 설정 검증 테스트")
    
    # 경로 확인
    for dataset_type in DatasetType:
        dataset_config = CONFIG.get_dataset_config(dataset_type)
        dataset_path = Path(CONFIG.data.namwon_data_path) / dataset_config["path"]
        
        if dataset_path.exists():
            logger.info(f"✅ {dataset_type.value}: {dataset_path}")
        else:
            logger.warning(f"⚠️ {dataset_type.value}: {dataset_path} (없음)")
    
    # 출력 경로 확인
    output_path = Path(CONFIG.data.output_path)
    logger.info(f"출력 경로: {output_path}")
    
    return True

if __name__ == "__main__":
    print("🏆 Nong-View Best Performance - 데이터 전처리 테스트")
    print("=" * 60)
    
    # 1. 설정 검증
    test_config_validation()
    
    # 2. 단일 데이터셋 테스트
    success = test_single_dataset()
    
    if success:
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n❌ 테스트 실패")