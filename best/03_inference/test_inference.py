#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 추론 시스템 테스트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from configs.best_config import CONFIG, logger
from optimized_inference import OptimizedInferenceEngine, MemoryMonitor, PerformanceTracker

def test_engine_initialization():
    """추론 엔진 초기화 테스트"""
    logger.info("🔧 추론 엔진 초기화 테스트")
    
    try:
        engine = OptimizedInferenceEngine(
            device="cpu",  # CPU로 테스트 (CUDA 없어도 동작)
            enable_half_precision=False,
            batch_size=2
        )
        
        logger.info("✅ 추론 엔진 초기화 성공")
        return engine
        
    except Exception as e:
        logger.error(f"❌ 추론 엔진 초기화 실패: {e}")
        return None

def test_memory_monitor():
    """메모리 모니터 테스트"""
    logger.info("💾 메모리 모니터 테스트")
    
    try:
        monitor = MemoryMonitor()
        
        current_usage = monitor.get_current_usage()
        available_memory = monitor.get_available_memory()
        peak_usage = monitor.get_peak_usage()
        
        logger.info(f"현재 메모리 사용량: {current_usage:.2f} MB")
        logger.info(f"사용 가능한 메모리: {available_memory:.2f} MB")
        logger.info(f"최대 메모리 사용량: {peak_usage:.2f} MB")
        
        logger.info("✅ 메모리 모니터 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 메모리 모니터 테스트 실패: {e}")
        return False

def test_performance_tracker():
    """성능 추적기 테스트"""
    logger.info("⚡ 성능 추적기 테스트")
    
    try:
        tracker = PerformanceTracker()
        
        # 더미 함수로 테스트
        @tracker.track_inference_time
        def dummy_inference():
            import time
            time.sleep(0.1)
            return "test_result"
        
        # 몇 번 실행
        for i in range(3):
            result = dummy_inference()
        
        summary = tracker.get_performance_summary()
        logger.info(f"성능 요약: {summary}")
        
        logger.info("✅ 성능 추적기 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 성능 추적기 테스트 실패: {e}")
        return False

def test_config_validation():
    """설정 검증 테스트"""
    logger.info("🔧 설정 검증 테스트")
    
    # 추론 설정 확인
    logger.info(f"추론 디바이스: {CONFIG.inference.device}")
    logger.info(f"배치 크기: {CONFIG.inference.batch_size}")
    logger.info(f"신뢰도 임계값: {CONFIG.inference.conf_threshold}")
    logger.info(f"IoU 임계값: {CONFIG.inference.iou_threshold}")
    logger.info(f"이미지 크기: {CONFIG.inference.image_size}")
    logger.info(f"반정밀도: {CONFIG.inference.half_precision}")
    
    # 출력 경로 확인
    output_path = Path(CONFIG.data.output_path)
    logger.info(f"출력 경로: {output_path}")
    
    return True

if __name__ == "__main__":
    print("🚀 Nong-View Best Performance - 추론 시스템 테스트")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    # 1. 설정 검증
    if test_config_validation():
        success_count += 1
    
    # 2. 메모리 모니터 테스트
    if test_memory_monitor():
        success_count += 1
    
    # 3. 성능 추적기 테스트
    if test_performance_tracker():
        success_count += 1
    
    # 4. 추론 엔진 초기화 테스트
    engine = test_engine_initialization()
    if engine:
        success_count += 1
        
        # 추가 정보 출력
        logger.info(f"모델 타입: {type(engine.model).__name__}")
        logger.info(f"디바이스: {engine.device}")
        logger.info(f"배치 크기: {engine.batch_size}")
        logger.info(f"반정밀도: {engine.enable_half_precision}")
    
    print("\n" + "=" * 60)
    print(f"테스트 결과: {success_count}/{total_tests} 성공")
    
    if success_count == total_tests:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")
        
    print("=" * 60)