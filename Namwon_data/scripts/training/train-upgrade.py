#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF 이미지 직접 학습 스크립트 (MBP 자동 최적화 버전 + CLI 지원)
고해상도 GeoTIFF 파일을 사용한 사료작물 탐지 모델 학습
+ MBP (Micro-Batch Processing): 큰 미니배치를 마이크로배치로 분할하여 VRAM 한계 극복
+ GMM 기반 전송 최적화: pinned memory, CUDA 최적화
+ Auto-Tuning: 네트워크 핑처럼 배치 크기를 자동으로 테스트하여 최적값 탐색
+ CLI 지원: 데이터셋 폴더명을 인자로 받아 유연하게 학습 가능

사용 예시:
    python train-upgrade.py --dataset dataset_greenhouse_multi --epochs 100
    python train-upgrade.py --dataset growth_tif_dataset --imgsz 1024 --auto-tune
"""

from ultralytics import YOLO
import torch
import os
import yaml
from pathlib import Path
from datetime import datetime
import warnings
import time
import argparse
import sys

# TIF 지원을 위한 환경 설정
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**31-1)  # OpenCV 픽셀 제한 해제
warnings.filterwarnings('ignore', category=UserWarning)  # TIF 경고 무시

# ============================================================================
# RTX A6000 48GB 하드코딩 설정
# ============================================================================
HARDWARE_CONFIG = {
    'vram_gb': 48,
    'gpu_name': 'RTX A6000',
    'cpu_cores': os.cpu_count() or 8,
}

DATASET_CONFIG = {
    'data_path': 'growth_tif_dataset/dataset.yaml',
    'imgsz': 1024,
}

TRAINING_CONFIG = {
    'epochs': 100,
    'device': 'cuda',
    'model': 'yolo11x-seg.pt',
    'project_name': 'growth_tif_training',
}

# 자동 튜닝 설정
AUTO_TUNE_CONFIG = {
    'enable': True,  # 자동 튜닝 활성화
    'start_micro': 2,  # 시작 마이크로배치 크기
    'start_target': 32,  # 시작 타깃 미니배치 크기
    'max_target': 128,  # 최대 타깃 미니배치 크기
    'increment_step': 32,  # 증가 단계 (32씩 증가)
    'test_iterations': 5,  # 테스트 반복 횟수 (안정성 확인)
    'safe_margin': 0.85,  # 안전 마진 (85% VRAM 사용까지만)
}

# ============================================================================
# TIF 4채널 → 3채널 변환 전처리
# ============================================================================
def convert_tif_to_3channel(data_path):
    """
    TIF 데이터셋의 4채널 이미지를 3채널로 변환
    """
    from PIL import Image
    import glob
    
    print("="*70)
    print("🔧 TIF 이미지 채널 변환 (4채널 → 3채널)")
    print("="*70)
    
    # dataset.yaml 읽기
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)
    
    # 이미지 경로 가져오기
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
        
        # TIF 파일 찾기
        tif_files = list(img_dir.glob('*.tif')) + list(img_dir.glob('*.tiff'))
        
        for tif_file in tif_files:
            total_checked += 1
            
            try:
                img = Image.open(tif_file)
                
                # 4채널 (RGBA) 체크
                if img.mode == 'RGBA' or img.n_frames > 3:
                    print(f"   🔄 변환: {tif_file.name} ({img.mode} → RGB)")
                    
                    # RGB로 변환
                    rgb_img = img.convert('RGB')
                    
                    # 원본 덮어쓰기
                    rgb_img.save(tif_file, compression='tiff_lzw')
                    
                    total_converted += 1
                    
                    img.close()
                    rgb_img.close()
                    
            except Exception as e:
                print(f"   ❌ 오류: {tif_file.name} - {e}")
    
    print(f"\n✅ 변환 완료!")
    print(f"   총 확인: {total_checked}개")
    print(f"   변환됨: {total_converted}개")
    
    if total_converted > 0:
        print(f"\n   ※ {total_converted}개 이미지가 3채널로 변환되었습니다.")
        print(f"   ※ 이제 YOLO 모델이 정상적으로 학습할 수 있습니다.")


# ============================================================================
# 배치 크기 자동 튜닝 함수 (네트워크 핑 방식)
# ============================================================================
def auto_tune_batch_size(data_path, imgsz, device, model_path):
    """
    네트워크 핑처럼 배치 크기를 점진적으로 테스트하여 최적값 찾기
    
    Returns:
        tuple: (최적_마이크로배치, 최적_타깃미니배치)
    """
    print("="*70)
    print("🔍 배치 크기 자동 튜닝 시작 (핑 테스트 방식)")
    print("="*70)
    
    if device != 'cuda' or not torch.cuda.is_available():
        print("⚠️ GPU 없음. 기본값 사용: micro=1, target=1")
        return 1, 1
    
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 VRAM 총량: {vram_total:.1f}GB")
    print(f"   안전 마진: {AUTO_TUNE_CONFIG['safe_margin']*100:.0f}% ({vram_total*AUTO_TUNE_CONFIG['safe_margin']:.1f}GB까지 사용)")
    
    # 모델 로드
    print(f"\n🤖 테스트용 모델 로드: {model_path}")
    model = YOLO(model_path)
    
    # 데이터셋 로드
    print(f"📂 데이터셋 로드: {data_path}")
    
    best_micro = AUTO_TUNE_CONFIG['start_micro']
    best_target = AUTO_TUNE_CONFIG['start_target']
    current_target = AUTO_TUNE_CONFIG['start_target']
    
    print(f"\n🎯 튜닝 시작점: micro={best_micro}, target={best_target}")
    print(f"   최대 목표: target={AUTO_TUNE_CONFIG['max_target']}")
    print(f"   증가 단계: {AUTO_TUNE_CONFIG['increment_step']}씩")
    
    test_results = []
    
    while current_target <= AUTO_TUNE_CONFIG['max_target']:
        accumulate = current_target // best_micro
        
        print(f"\n{'='*60}")
        print(f"📊 테스트 중: micro={best_micro} × {accumulate} = target={current_target}")
        print(f"{'='*60}")
        
        success = True
        max_vram_used = 0
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 짧은 테스트 학습 (몇 번의 iteration만)
            print(f"   반복 테스트 {AUTO_TUNE_CONFIG['test_iterations']}회 수행 중...")
            
            for i in range(AUTO_TUNE_CONFIG['test_iterations']):
                results = model.train(
                    data=data_path,
                    epochs=1,  # 1 에폭만
                    batch=best_micro,
                    imgsz=imgsz,
                    device=device,
                    nbs=current_target,  # nominal batch size로 자동 accumulate 계산
                    cache=False,
                    workers=2,  # 테스트는 워커 최소화
                    verbose=False,  # 로그 최소화
                    plots=False,
                    save=False,
                    exist_ok=True,
                    project='temp_tune',
                    name=f'test_{current_target}',
                    patience=0,
                    val=False,  # 검증 스킵
                )
                
                # VRAM 사용량 체크
                vram_used = torch.cuda.max_memory_allocated() / 1024**3
                max_vram_used = max(max_vram_used, vram_used)
                
                print(f"      [{i+1}/{AUTO_TUNE_CONFIG['test_iterations']}] VRAM: {vram_used:.2f}GB / {vram_total:.1f}GB ({vram_used/vram_total*100:.1f}%)")
                
                # 안전 마진 체크
                if vram_used > vram_total * AUTO_TUNE_CONFIG['safe_margin']:
                    print(f"      ⚠️ 안전 마진 초과! ({vram_used/vram_total*100:.1f}% > {AUTO_TUNE_CONFIG['safe_margin']*100:.0f}%)")
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
                
                # 성공했으므로 다음 단계로
                best_target = current_target
                current_target += AUTO_TUNE_CONFIG['increment_step']
            else:
                print(f"   ❌ 실패! 이전 설정으로 롤백")
                break
                
        except torch.cuda.OutOfMemoryError:
            print(f"   ❌ OOM 에러! 메모리 부족")
            test_results.append({
                'micro': best_micro,
                'target': current_target,
                'accumulate': accumulate,
                'success': False,
                'error': 'OOM'
            })
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
    
    # 결과 요약
    print("\n" + "="*70)
    print("📊 자동 튜닝 결과 요약")
    print("="*70)
    
    if test_results:
        print("\n성공한 테스트:")
        print(f"{'Target':>8} {'Micro':>8} {'Accumulate':>12} {'VRAM':>12} {'비율':>10}")
        print("-" * 60)
        for result in test_results:
            if result['success']:
                print(f"{result['target']:>8} {result['micro']:>8} {result['accumulate']:>12} "
                      f"{result['vram_used']:>10.2f}GB {result['vram_percent']:>9.1f}%")
    
    print(f"\n✨ 최적 설정 선택:")
    print(f"   마이크로배치 (micro): {best_micro}")
    print(f"   타깃 미니배치 (target): {best_target}")
    print(f"   그라디언트 누적 (accumulate): {best_target // best_micro}")
    
    expected_vram = test_results[-1]['vram_used'] if test_results and test_results[-1]['success'] else 0
    if expected_vram > 0:
        print(f"   예상 VRAM 사용량: {expected_vram:.2f}GB ({expected_vram/vram_total*100:.1f}%)")
    
    print(f"\n💡 튜닝 완료! 이제 전체 학습을 시작합니다.")
    
    time.sleep(2)  # 사용자가 결과를 확인할 시간
    return best_micro, best_target


# ============================================================================
# 메인 학습 함수
# ============================================================================
def train_tif_model_mbp(
    data_path,
    epochs,
    imgsz,
    device,
    model_path,
    project_name,
    mbp_micro,
    mbp_target
):
    """
    TIF 이미지를 직접 사용한 모델 학습 (MBP 최적화)
    """
    
    print("\n" + "="*70)
    print("🌱 사료작물 생육기 모델 학습 (TIF + MBP 최적화)")
    print("="*70)
    
    # === MBP 설정 ===
    accumulate = mbp_target // mbp_micro
    
    print("\n🧮 MBP (Micro-Batch Processing) 최종 설정")
    print(f"   마이크로배치 크기 (실제 batch): {mbp_micro}")
    print(f"   타깃 미니배치 크기: {mbp_target}")
    print(f"   그라디언트 누적(accumulate) 스텝: {accumulate}")
    print(f"   효과: VRAM은 micro={mbp_micro} 기준, 성능은 target={mbp_target} 기준")
    print(f"   ※ 각 마이크로배치 손실은 1/NSµ={1/accumulate:.4f}로 정규화됨")
    
    # === CUDA 최적화 설정 (GMM 아이디어 적용) ===
    if device == 'cuda' and torch.cuda.is_available():
        print("\n⚡ CUDA 전송 최적화 활성화")
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            print("   ✓ cuDNN benchmark 활성화")
        except Exception:
            pass
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("   ✓ TF32 활성화 (Ampere+ GPU 가속)")
        print("   ✓ Pinned memory: DataLoader에서 자동 활성화")
        print("   → CPU→GPU 전송 최적화로 I/O 병목 완화")
    
    # === 하드웨어 정보 ===
    print(f"\n🖥️ 하드웨어 정보")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {vram:.1f}GB")
        print(f"   CPU 코어: {HARDWARE_CONFIG['cpu_cores']}개")
    
    # === 데이터셋 정보 ===
    print(f"\n📂 데이터셋: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)
        print(f"   이미지 형식: TIF (GeoTIFF)")
        print(f"   이미지 크기: {dataset_info.get('img_size', [imgsz, imgsz])}")
        print(f"   클래스 수: {dataset_info['nc']}")
        print(f"   클래스: {dataset_info['names']}")
        
        if 'dataset_stats' in dataset_info:
            stats = dataset_info['dataset_stats']
            print(f"\n   📊 데이터 통계:")
            print(f"      Train: {stats.get('train', 'N/A')}개")
            print(f"      Val: {stats.get('val', 'N/A')}개")
            print(f"      Test: {stats.get('test', 'N/A')}개")
    
    # === 모델 초기화 ===
    print(f"\n🤖 모델 초기화: {model_path}")
    model = YOLO(model_path)
    
    # === Workers 동적 조정 ===
    workers = max(4, min(8, HARDWARE_CONFIG['cpu_cores'] // 2))
    if mbp_target >= 64:
        workers = max(6, workers)
    
    # === 학습 하이퍼파라미터 ===
    training_args = {
        'data': data_path,
        'epochs': epochs,
        'batch': mbp_micro,
        'imgsz': imgsz,
        'device': device,
        'project': project_name,
        'name': f'mbp_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # MBP 핵심 설정
        # Ultralytics는 nbs(nominal batch size)로 자동 accumulate 계산: accumulate = nbs / batch
        'nbs': mbp_target,  # nominal batch size = 타깃 미니배치 크기
        
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
        'patience': 30,
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
    
    print("\n📊 학습 하이퍼파라미터:")
    print(f"   에포크: {epochs}")
    print(f"   마이크로배치 (batch): {mbp_micro}")
    print(f"   타깃 미니배치 (nbs): {mbp_target}")
    print(f"   그라디언트 누적: {accumulate} 스텝 (자동 계산: nbs/batch = {mbp_target}/{mbp_micro})")
    print(f"   이미지 크기: {imgsz}x{imgsz}")
    print(f"   학습률 (lr0): {training_args['lr0']}")
    print(f"   옵티마이저: {training_args['optimizer']}")
    print(f"   AMP (혼합 정밀도): {training_args['amp']}")
    print(f"   DataLoader workers: {workers}")
    
    print("\n💡 MBP 작동 원리:")
    print(f"   1. {mbp_target}개 이미지를 {accumulate}번에 걸쳐 {mbp_micro}개씩 처리")
    print(f"   2. 각 마이크로배치마다 backward()로 그라디언트 누적")
    print(f"   3. {accumulate}번 누적 후 1회 optimizer.step() 실행")
    print(f"   4. 손실은 자동으로 1/{accumulate}={1/accumulate:.4f}로 정규화")
    print(f"   → VRAM은 {mbp_micro}개 기준, 성능은 {mbp_target}개 배치 효과!")
    print(f"   ※ Ultralytics가 nbs={mbp_target}와 batch={mbp_micro}로 자동 계산")
    
    # === 학습 시작 ===
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
        
        # 검증 결과
        if hasattr(results, 'results_dict'):
            print(f"\n📈 검증 결과:")
            metrics = results.results_dict
            if 'metrics/mAP50-95(M)' in metrics:
                print(f"   mAP@0.5-0.95 (Mask): {metrics['metrics/mAP50-95(M)']:.4f}")
            if 'metrics/mAP50(M)' in metrics:
                print(f"   mAP@0.5 (Mask): {metrics['metrics/mAP50(M)']:.4f}")
        
        print("\n📊 MBP 학습 분석 포인트:")
        print("   - 로그에서 optimizer step 빈도가 accumulate와 일치하는지 확인")
        print("   - loss scale이 안정적인지 확인")
        print("   - 일반 학습 대비 성능 비교 (mAP, IoU)")
        
        return best_model_path
    
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU 메모리 부족!")
        print("   자동 튜닝에서 선택한 설정도 실제 학습에서는 부족할 수 있습니다.")
        print("   해결: 코드에서 AUTO_TUNE_CONFIG['safe_margin']을 0.8 또는 0.75로 낮추세요.")
        return None
    
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# CLI 인자 파싱
# ============================================================================
def parse_args():
    """
    CLI 인자 파싱
    """
    parser = argparse.ArgumentParser(
        description='YOLOv11 Segmentation 학습 스크립트 (CLI 지원)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
사용 예시:
  python train-upgrade.py --dataset dataset_greenhouse_multi --epochs 100
  python train-upgrade.py --dataset growth_tif_dataset --imgsz 1024 --auto-tune
  python train-upgrade.py --dataset dataset_greenhouse_single --epochs 50 --model yolo11n-seg.pt
        '''
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='데이터셋 폴더 경로 (예: dataset_greenhouse_multi, growth_tif_dataset)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수 (기본값: 100)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='이미지 크기 (기본값: 1024)')
    parser.add_argument('--model', type=str, default='yolo11x-seg.pt',
                        help='YOLO 모델 파일 (기본값: yolo11x-seg.pt)')
    parser.add_argument('--project', type=str, default=None,
                        help='프로젝트 이름 (기본값: {dataset}_training)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스 (cuda 또는 cpu, 기본값: cuda)')
    parser.add_argument('--auto-tune', action='store_true',
                        help='배치 크기 자동 튜닝 활성화')
    parser.add_argument('--no-auto-tune', dest='auto_tune', action='store_false',
                        help='배치 크기 자동 튜닝 비활성화')
    parser.set_defaults(auto_tune=True)
    
    args = parser.parse_args()
    
    # 데이터셋 경로 검증
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ 오류: 데이터셋 폴더를 찾을 수 없습니다: {args.dataset}")
        sys.exit(1)
    
    # dataset.yaml 파일 확인
    yaml_path = dataset_path / 'dataset.yaml'
    if not yaml_path.exists():
        print(f"❌ 오류: dataset.yaml 파일을 찾을 수 없습니다: {yaml_path}")
        sys.exit(1)
    
    # 프로젝트 이름 자동 생성
    if args.project is None:
        args.project = f"{dataset_path.name}_training"
    
    return args


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    """
    메인 실행 함수
    """
    # CLI 인자 파싱
    args = parse_args()
    
    # 설정 적용
    data_path = str(Path(args.dataset) / 'dataset.yaml')
    
    print("="*70)
    print("🚀 TIF 이미지 직접 학습 시작 (MBP 자동 최적화 + CLI 지원)")
    print("="*70)
    print(f"\n📋 학습 설정:")
    print(f"   GPU: {HARDWARE_CONFIG['gpu_name']} ({HARDWARE_CONFIG['vram_gb']}GB)")
    print(f"   데이터셋: {args.dataset}")
    print(f"   YAML: {data_path}")
    print(f"   이미지 크기: {args.imgsz}x{args.imgsz}")
    print(f"   에포크: {args.epochs}")
    print(f"   모델: {args.model}")
    print(f"   프로젝트: {args.project}")
    print(f"   디바이스: {args.device}")
    print(f"\n   자동 튜닝: {'활성화 ✓' if args.auto_tune else '비활성화 ✗'}")
    
    if args.auto_tune:
        print(f"      시작점: micro={AUTO_TUNE_CONFIG['start_micro']}, target={AUTO_TUNE_CONFIG['start_target']}")
        print(f"      최대값: target={AUTO_TUNE_CONFIG['max_target']}")
        print(f"      증가폭: {AUTO_TUNE_CONFIG['increment_step']}")
        print(f"      안전마진: {AUTO_TUNE_CONFIG['safe_margin']*100:.0f}%")
    
    # GPU 체크
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n❌ CUDA를 사용할 수 없습니다!")
        return
    
    # TIF 4채널 → 3채널 변환 (학습 전 필수)
    print("\n")
    convert_tif_to_3channel(data_path)
    
    # 자동 튜닝으로 최적 배치 크기 찾기
    if args.auto_tune:
        mbp_micro, mbp_target = auto_tune_batch_size(
            data_path=data_path,
            imgsz=args.imgsz,
            device=args.device,
            model_path=args.model
        )
    else:
        # 자동 튜닝 비활성화 시 기본값
        mbp_micro = AUTO_TUNE_CONFIG['start_micro']
        mbp_target = AUTO_TUNE_CONFIG['start_target']
        print(f"\n📌 자동 튜닝 비활성화. 기본값 사용: micro={mbp_micro}, target={mbp_target}")
    
    # 실제 학습 시작
    train_tif_model_mbp(
        data_path=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        model_path=args.model,
        project_name=args.project,
        mbp_micro=mbp_micro,
        mbp_target=mbp_target
    )


if __name__ == "__main__":
    main()
