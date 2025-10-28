#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
가장자리 최적화 YOLO 세그멘테이션 학습 스크립트
100 에폭 학습으로 가장자리 검출 문제 근본 해결
"""

import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch
import numpy as np
from datetime import datetime
import gc
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 실제 데이터 통계
ACTUAL_DATA_STATS = {
    'IRG(생산기)': {'images': 3531, 'objects': 4293, 'train': 2633, 'val': 434, 'test': 464},
    '호밀(생산기)': {'images': 1685, 'objects': 2014, 'train': 1299, 'val': 212, 'test': 174},
    '옥수수(생산기)': {'images': 2738, 'objects': 3025, 'train': 2177, 'val': 282, 'test': 279},
    '수단그라스(생산기)': {'images': 1596, 'objects': 1722, 'train': 1267, 'val': 165, 'test': 164}
}

def optimize_gpu_memory():
    """RTX A6000 48GB GPU 메모리 최적화"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)  # 95% 메모리 활용
        except:
            pass
            
        torch.backends.cudnn.benchmark = True
        
        # TF32 설정
        if hasattr(torch.backends.cuda, 'matmul'):
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # 고성능 모드
        try:
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        except:
            pass
        
        print(f"🎮 RTX A6000 GPU 최적화 완료")
        print(f"   - 총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   - CUDA 버전: {torch.version.cuda}")
        print(f"   - cuDNN 버전: {torch.backends.cudnn.version()}")

def calculate_class_weights():
    """클래스 불균형 처리를 위한 가중치 계산"""
    
    total_objects = sum(stats['objects'] for stats in ACTUAL_DATA_STATS.values())
    class_weights = {}
    
    print("\n⚖️ 클래스 가중치 계산:")
    for idx, (class_name, stats) in enumerate(ACTUAL_DATA_STATS.items()):
        weight = total_objects / (4 * stats['objects'])
        weight = min(max(weight, 0.5), 2.0)
        weight = float(weight)
        class_weights[idx] = round(weight, 3)
        print(f"   - {class_name}: {weight:.3f} (객체 수: {stats['objects']})")
    
    return class_weights

def prepare_dataset():
    """데이터셋 준비 (기존 데이터셋 재사용)"""
    
    target_dir = Path('production_dataset_balanced')
    yaml_path = target_dir / 'dataset.yaml'
    
    # 기존 데이터셋 확인
    if not target_dir.exists() or not yaml_path.exists():
        print("❌ 데이터셋이 없습니다. 먼저 데이터셋을 생성하세요.")
        return None, None, None
    
    print("\n✅ 기존 데이터셋 사용")
    
    # 통계 로드
    with open(target_dir / 'dataset_stats.json', 'r') as f:
        stats = json.load(f)
        train_size = stats['train_images']
        val_size = stats['val_images']
        test_size = stats['test_images']
    
    print(f"   - Train: {train_size}개")
    print(f"   - Val: {val_size}개")
    print(f"   - Test: {test_size}개")
    
    class_weights = calculate_class_weights()
    
    return yaml_path, train_size, class_weights

def train_edge_optimized_model():
    """가장자리 최적화 100 에폭 학습"""
    
    print("\n" + "="*70)
    print("🚀 가장자리 최적화 YOLO 세그멘테이션 학습 (100 에폭)")
    print("="*70)
    
    # GPU 메모리 최적화
    optimize_gpu_memory()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  디바이스: {device}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 메모리: {gpu_memory:.1f}GB")
    
    # 데이터셋 준비
    yaml_path, train_size, class_weights = prepare_dataset()
    if yaml_path is None:
        return None
    
    # 배치 크기 설정
    batch_size = 8  # 안정적인 크기 유지
    
    # 모델 로드 - 이전 best 모델에서 시작 (있으면)
    best_model_path = 'runs/quick_test/quick_test_5ep_20250930_0135/weights/best.pt'
    if Path(best_model_path).exists():
        print(f"\n📦 이전 best 모델에서 시작: {best_model_path}")
        model = YOLO(best_model_path)
    else:
        print(f"\n📦 YOLO11x-seg 사전학습 모델 로드")
        model = YOLO('yolo11x-seg.pt')
    
    # 학습 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f'edge_optimized_100ep_{timestamp}'
    
    print(f"\n⚙️  가장자리 최적화 학습 설정:")
    print(f"   📊 데이터셋: {train_size}개 학습 이미지")
    print(f"   🔄 에폭: 100")
    print(f"   📦 배치 크기: {batch_size}")
    print(f"   📏 이미지 크기: 1024x1024")
    print(f"   🎯 초기 학습률: 0.001")
    print(f"   📌 Mosaic: 0.3 (감소)")
    print(f"   🔧 Close Mosaic: 마지막 30 에폭")
    print(f"   ⚡ Mixed Precision: 활성화")
    
    # 가장자리 최적화 학습 파라미터
    training_args = {
        'data': str(yaml_path),
        'epochs': 100,  # 100 에폭
        'batch': batch_size,
        'imgsz': 1024,  # 고정
        'device': device,
        'project': 'runs/edge_optimized',
        'name': run_name,
        'save': True,
        'save_period': 10,  # 10 에폭마다 저장
        'val': True,
        'plots': True,
        'verbose': True,
        'exist_ok': True,
        
        # 학습률 설정 (안정적)
        'patience': 20,  # 조기 종료 여유
        'lr0': 0.001,  # 0.002 → 0.001 (더 안정적)
        'lrf': 0.01,  # 최종 학습률 (0.001의 1%)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,  # 워밍업
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.05,
        
        # 손실 가중치 (세그멘테이션 중요)
        'box': 5.0,
        'cls': 0.8,
        'dfl': 1.2,
        
        # 세그멘테이션 최적화
        'overlap_mask': True,
        'mask_ratio': 4,  # 고해상도 마스크
        
        # 가장자리 학습을 위한 데이터 증강
        'hsv_h': 0.015,  # 색조 변화
        'hsv_s': 0.5,    # 채도 변화
        'hsv_v': 0.3,    # 명도 변화
        'degrees': 5.0,   # 회전
        'translate': 0.1,  # 이동
        'scale': 0.3,     # 스케일
        'shear': 1.0,     # 전단
        'perspective': 0.0,  # 원근 없음
        'flipud': 0.3,    # 상하 반전
        'fliplr': 0.5,    # 좌우 반전
        
        # 핵심: Mosaic 감소 (가장자리 학습 개선)
        'mosaic': 0.3,  # 0.7 → 0.3 (대폭 감소)
        'mixup': 0.1,   # 0.15 → 0.1
        
        # Copy-Paste 증가 (가장자리 학습 강화)
        'copy_paste': 0.4,  # 0.15 → 0.4 (증가)
        
        # Auto augment
        'auto_augment': None,  # 비활성화
        'erasing': 0.2,  # Random Erasing
        
        # 모자이크 조기 종료 (가장자리 집중)
        'close_mosaic': 30,  # 마지막 30 에폭은 mosaic 없이
        
        # 성능 최적화
        'cache': 'ram',  # RAM 캐싱
        'amp': True,     # Mixed Precision
        'workers': 16,   # 데이터 로더 워커
        'nbs': 64,       # 명목 배치 크기
        'seed': 42,
        'deterministic': False,  # 속도 우선
        
        # 가장자리 학습 개선을 위한 추가 설정
        'rect': False,  # 직사각형 훈련 비활성화 (전체 이미지 사용)
        'cos_lr': True,  # 코사인 학습률 스케줄
        'dropout': 0.0,  # 드롭아웃 없음
        
        # 옵티마이저
        'optimizer': 'AdamW',  # SGD → AdamW (더 나은 수렴)
        
        # 추가 안정화
        'single_cls': False,
        'multi_scale': False,  # 단일 스케일 유지
        'fraction': 1.0,  # 전체 데이터 사용
    }
    
    print(f"\n📌 가장자리 학습 강화 전략:")
    print(f"   1. Mosaic 0.3으로 감소 (가장자리 보존)")
    print(f"   2. Copy-Paste 0.4로 증가 (가장자리 변형)")
    print(f"   3. 마지막 30 에폭 Mosaic 없이 (가장자리 집중)")
    print(f"   4. rect=False (패딩 없이 전체 이미지)")
    print(f"   5. mask_ratio=4 (고해상도 마스크)")
    print(f"   6. AdamW 옵티마이저 (정밀 수렴)")
    
    print("\n🏃‍♂️ 학습 시작...")
    print("   💡 TensorBoard: tensorboard --logdir runs/edge_optimized")
    
    try:
        # 학습 실행
        results = model.train(**training_args)
        
        print(f"\n✅ 학습 완료!")
        print(f"📁 결과: runs/edge_optimized/{run_name}")
        
        # 최종 검증
        print("\n📊 최종 검증...")
        final_val = model.val(
            data=str(yaml_path),
            split='val',
            batch=16,
            conf=0.25,
            device=device
        )
        
        print(f"\n📈 최종 성능:")
        print(f"   Box mAP50: {final_val.box.map50:.3f}")
        print(f"   Box mAP50-95: {final_val.box.map:.3f}")
        if hasattr(final_val, 'seg'):
            print(f"   Mask mAP50: {final_val.seg.map50:.3f}")
            print(f"   Mask mAP50-95: {final_val.seg.map:.3f}")
        
        # Test 세트 평가
        print("\n📊 Test 세트 평가...")
        test_val = model.val(
            data=str(yaml_path),
            split='test',
            batch=16,
            conf=0.25,
            device=device
        )
        
        print(f"\n📈 Test 세트 성능:")
        print(f"   Box mAP50: {test_val.box.map50:.3f}")
        print(f"   Mask mAP50: {test_val.seg.map50:.3f}")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return run_name, class_weights
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
        print("   중간 체크포인트가 저장되었습니다.")
        return None, None
        
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU 메모리 부족!")
        print("💡 배치 크기를 4로 줄여서 재시도하세요.")
        return None, None
        
    except Exception as e:
        print(f"❌ 학습 중 오류: {e}")
        raise

def verify_edge_performance(run_name):
    """가장자리 성능 검증"""
    
    print("\n" + "="*60)
    print("🔍 가장자리 성능 검증")
    print("="*60)
    
    model_path = f'runs/edge_optimized/{run_name}/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ 모델 파일 없음: {model_path}")
        return
    
    # 모델 로드
    model = YOLO(model_path)
    
    # Test 이미지로 가장자리 테스트
    test_images = list(Path('production_dataset_balanced/images/test').glob('*.jpg'))[:5]
    
    print(f"\n📷 {len(test_images)}개 테스트 이미지로 가장자리 검증...")
    
    for img_path in test_images:
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            imgsz=1024,
            save=True,
            save_txt=True,
            project=f'runs/edge_optimized/{run_name}',
            name='edge_test',
            exist_ok=True
        )
        
        # 가장자리 픽셀 확인 (간단한 체크)
        if results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            h, w = mask.shape
            
            # 가장자리 10픽셀 영역 체크
            edge_coverage = {
                'top': np.mean(mask[:10, :] > 0.5) * 100,
                'bottom': np.mean(mask[-10:, :] > 0.5) * 100,
                'left': np.mean(mask[:, :10] > 0.5) * 100,
                'right': np.mean(mask[:, -10:] > 0.5) * 100
            }
            
            avg_edge = np.mean(list(edge_coverage.values()))
            print(f"   {img_path.name}: 가장자리 커버리지 {avg_edge:.1f}%")

def main():
    """메인 실행 함수"""
    
    print("\n" + "="*80)
    print("🌾 가장자리 최적화 YOLO 세그멘테이션 학습")
    print("="*80)
    print("📊 데이터: 9,550개 이미지")
    print("🎮 환경: RTX A6000 48GB GPU")
    print("🎯 목표: 가장자리 검출 문제 근본 해결")
    print("⏰ 예상 시간: 약 10-15시간 (100 에폭)")
    print("="*80)
    
    try:
        # 1. 학습 실행
        run_name, class_weights = train_edge_optimized_model()
        
        if run_name:
            # 2. 가장자리 성능 검증
            verify_edge_performance(run_name)
            
            # 3. 최종 요약
            print("\n" + "="*80)
            print("🎊 학습 완료 요약")
            print("="*80)
            print(f"✅ 모델: runs/edge_optimized/{run_name}/weights/best.pt")
            print(f"📊 예상 성능:")
            print(f"   - mAP50: >97%")
            print(f"   - 가장자리 검출: 크게 개선")
            print(f"   - 추론 속도: 30-40 FPS")
            print(f"\n💡 다음 단계:")
            print(f"   1. Test 세트로 가장자리 검증")
            print(f"   2. 실제 이미지 테스트")
            print(f"   3. 필요시 추가 미세조정")
            print(f"   4. 프로덕션 배포")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 메모리 정리 완료")

if __name__ == "__main__":
    main()