#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클래스별 YOLOv11 Segmentation 모델 학습 스크립트
비닐하우스 단동, 비닐하우스 다동, 곤포사일리지 각각 별도 모델 학습
"""

from ultralytics import YOLO
import torch
import os
import yaml
from pathlib import Path
from datetime import datetime
import warnings
import time

# TIF 지원을 위한 환경 설정
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**31-1)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# 데이터셋 설정
# ============================================================================
DATASETS = {
    'greenhouse_single': {
        'name': '비닐하우스_단동',
        'data_path': 'dataset_greenhouse_single/dataset.yaml',
        'project_name': 'greenhouse_single_training',
    },
    'greenhouse_multi': {
        'name': '비닐하우스_다동',
        'data_path': 'dataset_greenhouse_multi/dataset.yaml',
        'project_name': 'greenhouse_multi_training',
    },
    'silage_bale': {
        'name': '곤포사일리지',
        'data_path': 'dataset_silage_bale/dataset.yaml',
        'project_name': 'silage_bale_training',
    }
}

# ============================================================================
# 하드웨어 설정
# ============================================================================
HARDWARE_CONFIG = {
    'vram_gb': 48,
    'gpu_name': 'RTX A6000',
    'cpu_cores': os.cpu_count() or 8,
}

# ============================================================================
# 학습 설정
# ============================================================================
TRAINING_CONFIG = {
    'epochs': 100,
    'device': 'cuda',
    'model': 'yolo11x-seg.pt',
    'imgsz': 1024,
    'patience': 30,
}

# 자동 튜닝 설정
AUTO_TUNE_CONFIG = {
    'enable': True,
    'start_micro': 2,
    'start_target': 32,
    'max_target': 128,
    'increment_step': 32,
    'test_iterations': 5,
    'safe_margin': 0.85,
}

# ============================================================================
# TIF 4채널 → 3채널 변환
# ============================================================================
def convert_tif_to_3channel(data_path):
    """TIF 데이터셋의 4채널 이미지를 3채널로 변환"""
    from PIL import Image

    print("="*70)
    print("🔧 TIF 이미지 채널 변환 (4채널 → 3채널)")
    print("="*70)

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)

    dataset_dir = Path(data_path).parent
    image_dirs = []

    if 'train' in dataset_info:
        image_dirs.append(dataset_dir / dataset_info['train'])
    if 'val' in dataset_info:
        image_dirs.append(dataset_dir / dataset_info['val'])
    if 'test' in dataset_info:
        image_dirs.append(dataset_dir / dataset_info['test'])

    total_converted = 0
    total_checked = 0

    for img_dir in image_dirs:
        if not img_dir.exists():
            continue

        print(f"\n📂 처리 중: {img_dir}")

        tif_files = list(img_dir.glob('*.tif')) + list(img_dir.glob('*.tiff'))

        for tif_file in tif_files:
            total_checked += 1

            try:
                img = Image.open(tif_file)

                if img.mode == 'RGBA' or (hasattr(img, 'n_frames') and img.n_frames > 3):
                    print(f"   🔄 변환: {tif_file.name} ({img.mode} → RGB)")

                    rgb_img = img.convert('RGB')
                    rgb_img.save(tif_file, compression='tiff_lzw')

                    total_converted += 1
                    img.close()
                    rgb_img.close()

            except Exception as e:
                print(f"   ⚠️  오류: {tif_file.name} - {e}")

    print(f"\n✅ 변환 완료: {total_checked}개 확인, {total_converted}개 변환")

# ============================================================================
# 배치 크기 자동 튜닝
# ============================================================================
def auto_tune_batch_size(data_path, imgsz, device, model_path):
    """배치 크기 자동 튜닝"""

    print("="*70)
    print("🔍 배치 크기 자동 튜닝 시작")
    print("="*70)

    if device != 'cuda' or not torch.cuda.is_available():
        print("⚠️ GPU 없음. 기본값 사용: micro=1, target=1")
        return 1, 1

    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 VRAM 총량: {vram_total:.1f}GB")
    print(f"   안전 마진: {AUTO_TUNE_CONFIG['safe_margin']*100:.0f}%")

    print(f"\n🤖 테스트용 모델 로드: {model_path}")
    model = YOLO(model_path)

    best_micro = AUTO_TUNE_CONFIG['start_micro']
    best_target = AUTO_TUNE_CONFIG['start_target']
    current_target = AUTO_TUNE_CONFIG['start_target']

    print(f"\n🎯 튜닝 시작점: micro={best_micro}, target={best_target}")

    test_results = []

    while current_target <= AUTO_TUNE_CONFIG['max_target']:
        accumulate = current_target // best_micro

        print(f"\n{'='*60}")
        print(f"📊 테스트 중: micro={best_micro} × {accumulate} = target={current_target}")

        success = True
        max_vram_used = 0

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print(f"   반복 테스트 {AUTO_TUNE_CONFIG['test_iterations']}회 수행 중...")

            for i in range(AUTO_TUNE_CONFIG['test_iterations']):
                results = model.train(
                    data=data_path,
                    epochs=1,
                    batch=best_micro,
                    imgsz=imgsz,
                    device=device,
                    nbs=current_target,
                    cache=False,
                    workers=2,
                    verbose=False,
                    plots=False,
                    save=False,
                    exist_ok=True,
                    project='temp_tune',
                    name=f'test_{current_target}',
                    patience=0,
                    val=False,
                )

                vram_used = torch.cuda.max_memory_allocated() / 1024**3
                max_vram_used = max(max_vram_used, vram_used)

                print(f"      [{i+1}/{AUTO_TUNE_CONFIG['test_iterations']}] VRAM: {vram_used:.2f}GB / {vram_total:.1f}GB ({vram_used/vram_total*100:.1f}%)")

                if vram_used > vram_total * AUTO_TUNE_CONFIG['safe_margin']:
                    print(f"      ⚠️ 안전 마진 초과!")
                    success = False
                    break

                torch.cuda.empty_cache()
                time.sleep(0.5)

            if success:
                vram_percent = max_vram_used / vram_total * 100
                print(f"   ✅ 성공! 최대 VRAM: {max_vram_used:.2f}GB ({vram_percent:.1f}%)")

                test_results.append({
                    'micro': best_micro,
                    'target': current_target,
                    'accumulate': accumulate,
                    'vram_used': max_vram_used,
                    'vram_percent': vram_percent,
                    'success': True
                })

                best_target = current_target
                current_target += AUTO_TUNE_CONFIG['increment_step']
            else:
                print(f"   ❌ 실패! 이전 설정으로 롤백")
                break

        except torch.cuda.OutOfMemoryError:
            print(f"   ❌ OOM 에러! 메모리 부족")
            break

        except Exception as e:
            print(f"   ❌ 에러 발생: {e}")
            break

    # 임시 파일 정리
    print("\n🧹 테스트 파일 정리 중...")
    try:
        import shutil
        if os.path.exists('temp_tune'):
            shutil.rmtree('temp_tune')
    except:
        pass

    torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("📊 자동 튜닝 결과")
    print("="*70)

    if test_results:
        print("\n성공한 테스트:")
        print(f"{'Target':>8} {'Micro':>8} {'Accumulate':>12} {'VRAM':>12} {'비율':>10}")
        print("-" * 60)
        for result in test_results:
            if result['success']:
                print(f"{result['target']:>8} {result['micro']:>8} {result['accumulate']:>12} "
                      f"{result['vram_used']:>10.2f}GB {result['vram_percent']:>9.1f}%")

    print(f"\n✨ 최적 설정:")
    print(f"   마이크로배치: {best_micro}")
    print(f"   타깃 미니배치: {best_target}")
    print(f"   그라디언트 누적: {best_target // best_micro}")

    time.sleep(2)
    return best_micro, best_target

# ============================================================================
# 메인 학습 함수
# ============================================================================
def train_model(dataset_key, mbp_micro, mbp_target):
    """모델 학습"""

    dataset_config = DATASETS[dataset_key]
    data_path = dataset_config['data_path']
    project_name = dataset_config['project_name']

    print("\n" + "="*70)
    print(f"🌱 {dataset_config['name']} 모델 학습 시작")
    print("="*70)

    accumulate = mbp_target // mbp_micro

    print("\n🧮 MBP (Micro-Batch Processing) 설정")
    print(f"   마이크로배치 크기: {mbp_micro}")
    print(f"   타깃 미니배치 크기: {mbp_target}")
    print(f"   그라디언트 누적 스텝: {accumulate}")

    # CUDA 최적화
    if TRAINING_CONFIG['device'] == 'cuda' and torch.cuda.is_available():
        print("\n⚡ CUDA 최적화 활성화")
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   ✓ cuDNN benchmark, TF32 활성화")
        except:
            pass

        print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {vram:.1f}GB")

    # 데이터셋 정보
    print(f"\n📂 데이터셋: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)
        print(f"   클래스 수: {dataset_info['nc']}")
        print(f"   클래스: {dataset_info['names']}")

        if 'dataset_stats' in dataset_info:
            stats = dataset_info['dataset_stats']
            print(f"\n   📊 데이터 통계:")
            print(f"      Train: {stats.get('train_images', 'N/A')}개")
            print(f"      Val: {stats.get('val_images', 'N/A')}개")
            print(f"      Test: {stats.get('test_images', 'N/A')}개")
            print(f"      총 객체: {stats.get('total_objects', 'N/A')}개")

    # 모델 초기화
    print(f"\n🤖 모델 초기화: {TRAINING_CONFIG['model']}")
    model = YOLO(TRAINING_CONFIG['model'])

    # Workers 동적 조정
    workers = max(4, min(8, HARDWARE_CONFIG['cpu_cores'] // 2))
    if mbp_target >= 64:
        workers = max(6, workers)

    # 학습 하이퍼파라미터
    training_args = {
        'data': data_path,
        'epochs': TRAINING_CONFIG['epochs'],
        'batch': mbp_micro,
        'imgsz': TRAINING_CONFIG['imgsz'],
        'device': TRAINING_CONFIG['device'],
        'project': project_name,
        'name': f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

        # MBP 설정
        'nbs': mbp_target,

        # 옵티마이저
        'optimizer': 'AdamW',
        'lr0': 0.0008,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 증강 설정
        'hsv_h': 0.010,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 5,
        'translate': 0.05,
        'scale': 0.3,
        'shear': 1.0,
        'perspective': 0.0,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.2,
        'mixup': 0.05,
        'copy_paste': 0.5,
        'close_mosaic': 40,

        # 학습 설정
        'patience': TRAINING_CONFIG['patience'],
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': workers,
        'exist_ok': True,
        'pretrained': True,
        'amp': True,
        'val': True,
        'plots': True,

        # 손실 가중치
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,

        # 기타
        'rect': False,
        'cos_lr': True,
    }

    print("\n📊 학습 설정:")
    print(f"   에포크: {TRAINING_CONFIG['epochs']}")
    print(f"   이미지 크기: {TRAINING_CONFIG['imgsz']}x{TRAINING_CONFIG['imgsz']}")
    print(f"   학습률: {training_args['lr0']}")
    print(f"   옵티마이저: {training_args['optimizer']}")
    print(f"   DataLoader workers: {workers}")

    print("\n" + "="*70)
    print("🚀 학습 시작!")
    print("="*70)

    try:
        torch.cuda.empty_cache()

        results = model.train(**training_args)

        print("\n" + "="*70)
        print("✅ 학습 완료!")
        print("="*70)

        # 모델 경로
        best_model_path = Path(project_name) / training_args['name'] / 'weights' / 'best.pt'
        last_model_path = Path(project_name) / training_args['name'] / 'weights' / 'last.pt'

        print(f"\n📁 모델 저장 위치:")
        print(f"   최고 성능: {best_model_path}")
        print(f"   마지막: {last_model_path}")

        return best_model_path

    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU 메모리 부족!")
        print("   해결: AUTO_TUNE_CONFIG['safe_margin']을 낮추세요.")
        return None

    except Exception as e:
        print(f"\n❌ 학습 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 메인 실행
# ============================================================================
def main():
    print("="*70)
    print("🚀 클래스별 YOLOv11 Segmentation 모델 학습")
    print("="*70)

    print(f"\n📋 학습할 모델: {len(DATASETS)}개")
    for key, config in DATASETS.items():
        print(f"   - {config['name']}: {config['data_path']}")

    print(f"\n⚙️  학습 설정:")
    print(f"   에포크: {TRAINING_CONFIG['epochs']}")
    print(f"   이미지 크기: {TRAINING_CONFIG['imgsz']}")
    print(f"   모델: {TRAINING_CONFIG['model']}")
    print(f"   자동 튜닝: {'활성화 ✓' if AUTO_TUNE_CONFIG['enable'] else '비활성화 ✗'}")

    # GPU 체크
    if not torch.cuda.is_available():
        print("\n❌ CUDA를 사용할 수 없습니다!")
        return

    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   VRAM: {vram:.1f}GB")

    # 각 데이터셋별로 순차 학습
    results = {}

    for dataset_key, dataset_config in DATASETS.items():
        print("\n" + "="*70)
        print(f"📦 {dataset_config['name']} 처리 중...")
        print("="*70)

        data_path = dataset_config['data_path']

        # TIF 채널 변환
        print("\n")
        convert_tif_to_3channel(data_path)

        # 자동 튜닝
        if AUTO_TUNE_CONFIG['enable']:
            mbp_micro, mbp_target = auto_tune_batch_size(
                data_path=data_path,
                imgsz=TRAINING_CONFIG['imgsz'],
                device=TRAINING_CONFIG['device'],
                model_path=TRAINING_CONFIG['model']
            )
        else:
            mbp_micro = AUTO_TUNE_CONFIG['start_micro']
            mbp_target = AUTO_TUNE_CONFIG['start_target']

        # 학습 시작
        best_model = train_model(dataset_key, mbp_micro, mbp_target)
        results[dataset_key] = best_model

        print(f"\n✅ {dataset_config['name']} 완료!")
        if best_model:
            print(f"   모델: {best_model}")

    # 최종 요약
    print("\n" + "="*70)
    print("🎉 전체 학습 완료!")
    print("="*70)

    for key, model_path in results.items():
        dataset_name = DATASETS[key]['name']
        if model_path:
            print(f"\n✅ {dataset_name}")
            print(f"   📁 {model_path}")
        else:
            print(f"\n❌ {dataset_name}: 학습 실패")

if __name__ == "__main__":
    main()
