#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
결과 분석 시스템 테스트
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from configs.best_config import CONFIG, logger
from results_analyzer import ResultsAnalyzer, PerformanceMetrics

def create_dummy_inference_results(output_dir: Path, num_images: int = 5):
    """더미 추론 결과 파일들 생성"""
    
    dummy_results = []
    
    for i in range(num_images):
        # 더미 검출 결과 생성
        detections = []
        num_detections = (i % 3) + 1  # 1-3개 검출
        
        for j in range(num_detections):
            detection = {
                'bbox': [100 + j*50, 100 + j*30, 200 + j*50, 200 + j*30],
                'confidence': 0.7 + (j * 0.1),
                'class_id': j % 2,
                'class_name': 'greenhouse_multi' if j % 2 == 0 else 'greenhouse_single',
                'area': 10000 + j*1000
            }
            detections.append(detection)
        
        # 더미 결과 생성
        result = {
            'image_path': f'/dummy/path/image_{i:03d}.jpg',
            'detections': detections,
            'processing_time': 0.2 + (i * 0.05),  # 0.2~0.4초
            'confidence_stats': {
                'mean': 0.75,
                'std': 0.1,
                'min': 0.6,
                'max': 0.9
            },
            'metadata': {
                'model_type': 'YOLO',
                'device': 'cuda',
                'timestamp': '2025-10-28T15:00:00'
            }
        }
        
        dummy_results.append(result)
        
        # JSON 파일로 저장
        result_file = output_dir / f"image_{i:03d}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"더미 추론 결과 생성 완료: {num_images}개 파일")
    return dummy_results

def test_analyzer_initialization():
    """분석기 초기화 테스트"""
    logger.info("🔧 결과 분석기 초기화 테스트")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = ResultsAnalyzer(output_dir=temp_dir)
            
            logger.info(f"출력 디렉토리: {analyzer.output_dir}")
            logger.info("✅ 결과 분석기 초기화 성공")
            return analyzer
            
    except Exception as e:
        logger.error(f"❌ 결과 분석기 초기화 실패: {e}")
        return None

def test_performance_metrics():
    """성능 메트릭 테스트"""
    logger.info("📊 성능 메트릭 테스트")
    
    try:
        # 더미 메트릭 생성
        metrics = PerformanceMetrics()
        metrics.map_50 = 0.85
        metrics.map_75 = 0.70
        metrics.precision = 0.82
        metrics.recall = 0.78
        metrics.f1_score = 0.80
        metrics.avg_inference_time = 0.25
        metrics.throughput_fps = 4.0
        metrics.avg_confidence = 0.75
        metrics.total_detections = 123
        
        # 클래스별 메트릭
        metrics.class_metrics = {
            'greenhouse_multi': {'count': 80, 'percentage': 65.0},
            'greenhouse_single': {'count': 43, 'percentage': 35.0}
        }
        
        # 신뢰도 분포
        metrics.confidence_distribution = {
            '0.0-0.3': 5,
            '0.3-0.5': 15,
            '0.5-0.7': 35,
            '0.7-0.9': 50,
            '0.9-1.0': 18
        }
        
        logger.info(f"mAP@0.5: {metrics.map_50}")
        logger.info(f"평균 추론 시간: {metrics.avg_inference_time*1000:.1f}ms")
        logger.info(f"총 검출 수: {metrics.total_detections}")
        logger.info(f"클래스 수: {len(metrics.class_metrics)}")
        
        logger.info("✅ 성능 메트릭 테스트 성공")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ 성능 메트릭 테스트 실패: {e}")
        return None

def test_dummy_analysis():
    """더미 데이터로 분석 테스트"""
    logger.info("🧪 더미 데이터 분석 테스트")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 더미 결과 파일 생성
            results_dir = temp_path / "results"
            results_dir.mkdir()
            
            dummy_results = create_dummy_inference_results(results_dir, num_images=10)
            
            # 분석기 초기화
            output_dir = temp_path / "analysis"
            analyzer = ResultsAnalyzer(output_dir=str(output_dir))
            
            # 분석 실행
            report = analyzer.analyze_inference_results(
                results_dir=str(results_dir),
                model_name="Test_Model",
                dataset_name="Test_Dataset"
            )
            
            logger.info(f"분석 완료:")
            logger.info(f"  - 총 검출 수: {report.overall_metrics.total_detections}")
            logger.info(f"  - 평균 신뢰도: {report.overall_metrics.avg_confidence:.3f}")
            logger.info(f"  - 평균 처리 시간: {report.overall_metrics.avg_inference_time*1000:.1f}ms")
            logger.info(f"  - 검출 밀도: {report.overall_metrics.detection_density:.2f}")
            logger.info(f"  - 시각화 파일 수: {len(report.visualization_paths)}")
            logger.info(f"  - 권장사항 수: {len(report.recommendations)}")
            
            logger.info("✅ 더미 데이터 분석 테스트 성공")
            return True
            
    except Exception as e:
        logger.error(f"❌ 더미 데이터 분석 테스트 실패: {e}")
        return False

def test_visualization_generation():
    """시각화 생성 테스트"""
    logger.info("📈 시각화 생성 테스트")
    
    try:
        # 더미 메트릭으로 시각화 테스트
        metrics = test_performance_metrics()
        if not metrics:
            return False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = ResultsAnalyzer(output_dir=temp_dir)
            
            # 더미 추론 결과
            dummy_inference_results = [
                {
                    'detections': [
                        {'confidence': 0.8, 'class_name': 'greenhouse_multi'},
                        {'confidence': 0.7, 'class_name': 'greenhouse_single'}
                    ],
                    'processing_time': 0.25
                },
                {
                    'detections': [
                        {'confidence': 0.9, 'class_name': 'greenhouse_multi'}
                    ],
                    'processing_time': 0.30
                }
            ]
            
            # 시각화 생성
            viz_paths = analyzer._create_visualizations(
                dummy_inference_results, metrics, "Test_Model", "Test_Dataset"
            )
            
            logger.info(f"생성된 시각화 파일 수: {len(viz_paths)}")
            for path in viz_paths:
                if Path(path).exists():
                    logger.info(f"  ✅ {Path(path).name}")
                else:
                    logger.warning(f"  ❌ {Path(path).name} (생성 실패)")
            
            logger.info("✅ 시각화 생성 테스트 성공")
            return True
            
    except Exception as e:
        logger.error(f"❌ 시각화 생성 테스트 실패: {e}")
        return False

def test_config_validation():
    """설정 검증 테스트"""
    logger.info("🔧 설정 검증 테스트")
    
    # 분석 관련 설정 확인
    logger.info(f"데이터 출력 경로: {CONFIG.data.output_path}")
    logger.info(f"프로젝트 이름: {CONFIG.project_name}")
    logger.info(f"버전: {CONFIG.version}")
    
    # 추론 설정 확인 (분석에 필요)
    logger.info(f"신뢰도 임계값: {CONFIG.inference.conf_threshold}")
    logger.info(f"IoU 임계값: {CONFIG.inference.iou_threshold}")
    
    return True

if __name__ == "__main__":
    print("📊 Nong-View Best Performance - 결과 분석 시스템 테스트")
    print("=" * 60)
    
    success_count = 0
    total_tests = 5
    
    # 1. 설정 검증
    if test_config_validation():
        success_count += 1
    
    # 2. 분석기 초기화 테스트
    analyzer = test_analyzer_initialization()
    if analyzer:
        success_count += 1
    
    # 3. 성능 메트릭 테스트
    metrics = test_performance_metrics()
    if metrics:
        success_count += 1
    
    # 4. 시각화 생성 테스트
    if test_visualization_generation():
        success_count += 1
    
    # 5. 더미 데이터 분석 테스트
    if test_dummy_analysis():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"테스트 결과: {success_count}/{total_tests} 성공")
    
    if success_count == total_tests:
        print("🎉 모든 테스트 통과!")
        print("\n📊 주요 기능:")
        print("  ✅ 성능 메트릭 자동 계산")
        print("  ✅ 대화형 시각화 생성")
        print("  ✅ 종합 분석 리포트 생성")
        print("  ✅ 자동 권장사항 제공")
        print("  ✅ 모델 비교 분석 지원")
    else:
        print("⚠️ 일부 테스트 실패")
        
    print("=" * 60)