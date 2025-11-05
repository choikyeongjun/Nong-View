"""
Segmentation Configuration Test Script
Segmentation 설정이 제대로 작동하는지 검증
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.best_config import ModelType, DatasetType, CONFIG
from optimized_training import (
    create_training_config,
    TrainingStrategy,
    TrainingConfig
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_types():
    """모델 타입 테스트"""
    logger.info("\n" + "=" * 60)
    logger.info("1. 모델 타입 테스트")
    logger.info("=" * 60)
    
    # Detection 모델
    logger.info("\n[Detection 모델]")
    for model in [ModelType.YOLO11N, ModelType.YOLO11S, ModelType.YOLO11M]:
        logger.info(f"  - {model.name}: {model.value}")
    
    # Segmentation 모델
    logger.info("\n[Segmentation 모델]")
    for model in [ModelType.YOLO11N_SEG, ModelType.YOLO11S_SEG, ModelType.YOLO11M_SEG]:
        logger.info(f"  - {model.name}: {model.value}")
    
    logger.info("✓ 모델 타입 테스트 통과")


def test_dataset_types():
    """데이터셋 타입 테스트"""
    logger.info("\n" + "=" * 60)
    logger.info("2. 데이터셋 타입 테스트")
    logger.info("=" * 60)
    
    # Detection 데이터셋
    logger.info("\n[Detection 데이터셋]")
    for dataset in [DatasetType.GREENHOUSE_MULTI, DatasetType.GREENHOUSE_SINGLE, DatasetType.GROWTH_TIF]:
        logger.info(f"  - {dataset.name}: {dataset.value}")
    
    # Segmentation 데이터셋
    logger.info("\n[Segmentation 데이터셋]")
    dataset = DatasetType.MODEL3_GREENHOUSE_SEG
    logger.info(f"  - {dataset.name}: {dataset.value}")
    
    # 데이터셋 정보 확인
    dataset_info = CONFIG.data.dataset_info[DatasetType.MODEL3_GREENHOUSE_SEG]
    logger.info(f"\n[MODEL3_GREENHOUSE_SEG 정보]")
    logger.info(f"  - Path: {dataset_info['path']}")
    logger.info(f"  - Classes: {dataset_info['classes']}")
    logger.info(f"  - Total Images: {dataset_info['total_images']}")
    logger.info(f"  - Task: {dataset_info.get('task', 'detect')}")
    
    logger.info("✓ 데이터셋 타입 테스트 통과")


def test_segmentation_config():
    """Segmentation 설정 테스트"""
    logger.info("\n" + "=" * 60)
    logger.info("3. Segmentation 설정 생성 테스트")
    logger.info("=" * 60)
    
    config = create_training_config(
        model_type=ModelType.YOLO11N_SEG,
        dataset_type=DatasetType.MODEL3_GREENHOUSE_SEG,
        strategy=TrainingStrategy.PROGRESSIVE
    )
    
    logger.info(f"\n[기본 설정]")
    logger.info(f"  - Model Type: {config.model_type.name}")
    logger.info(f"  - Dataset Type: {config.dataset_type.name}")
    logger.info(f"  - Task: {config.task}")
    logger.info(f"  - Epochs: {config.epochs}")
    logger.info(f"  - Batch Size: {config.batch_size}")
    logger.info(f"  - Image Size: {config.imgsz}")
    logger.info(f"  - Base LR: {config.base_lr}")
    logger.info(f"  - Strategy: {config.strategy.value}")
    
    logger.info(f"\n[손실 가중치]")
    logger.info(f"  - Box Loss Weight: {config.box_loss_weight}")
    logger.info(f"  - Cls Loss Weight: {config.cls_loss_weight}")
    logger.info(f"  - DFL Loss Weight: {config.dfl_loss_weight}")
    logger.info(f"  - Mask Loss Weight: {config.mask_loss_weight}")
    
    logger.info(f"\n[Segmentation 전용 설정]")
    logger.info(f"  - Overlap Mask: {config.overlap_mask}")
    logger.info(f"  - Mask Ratio: {config.mask_ratio}")
    
    logger.info(f"\n[데이터 증강]")
    logger.info(f"  - Mosaic: {config.mosaic}")
    logger.info(f"  - Mixup: {config.mixup}")
    logger.info(f"  - Copy Paste: {config.copy_paste}")
    logger.info(f"  - Degrees: {config.degrees}")
    logger.info(f"  - Scale: {config.scale}")
    
    # 검증
    assert config.task == 'segment', "Task should be 'segment'"
    assert config.overlap_mask == True, "overlap_mask should be True"
    assert config.mask_ratio == 4, "mask_ratio should be 4"
    assert hasattr(config, 'mask_loss_weight'), "Should have mask_loss_weight"
    
    logger.info("\n✓ Segmentation 설정 테스트 통과")


def test_detection_config():
    """Detection 설정 테스트 (호환성 확인)"""
    logger.info("\n" + "=" * 60)
    logger.info("4. Detection 설정 테스트 (호환성)")
    logger.info("=" * 60)
    
    config = create_training_config(
        model_type=ModelType.YOLO11N,
        dataset_type=DatasetType.GREENHOUSE_SINGLE,
        strategy=TrainingStrategy.PROGRESSIVE
    )
    
    logger.info(f"\n[기본 설정]")
    logger.info(f"  - Model Type: {config.model_type.name}")
    logger.info(f"  - Dataset Type: {config.dataset_type.name}")
    logger.info(f"  - Task: {config.task}")
    logger.info(f"  - Epochs: {config.epochs}")
    logger.info(f"  - Batch Size: {config.batch_size}")
    
    # 검증
    assert config.task == 'detect', "Task should be 'detect'"
    
    logger.info("\n✓ Detection 설정 테스트 통과 (기존 기능 유지)")


def test_all_seg_models():
    """모든 Segmentation 모델 설정 테스트"""
    logger.info("\n" + "=" * 60)
    logger.info("5. 모든 Segmentation 모델 설정 테스트")
    logger.info("=" * 60)
    
    seg_models = [
        ModelType.YOLO11N_SEG,
        ModelType.YOLO11S_SEG,
        ModelType.YOLO11M_SEG,
        ModelType.YOLO11L_SEG,
        ModelType.YOLO11X_SEG
    ]
    
    for model in seg_models:
        config = create_training_config(
            model_type=model,
            dataset_type=DatasetType.MODEL3_GREENHOUSE_SEG,
            strategy=TrainingStrategy.PROGRESSIVE
        )
        
        logger.info(f"\n[{model.name}]")
        logger.info(f"  - Task: {config.task}")
        logger.info(f"  - Batch Size: {config.batch_size}")
        logger.info(f"  - Base LR: {config.base_lr}")
        logger.info(f"  - Overlap Mask: {config.overlap_mask}")
        logger.info(f"  - Mask Ratio: {config.mask_ratio}")
        
        assert config.task == 'segment'
        assert config.overlap_mask == True
        assert config.mask_ratio == 4
    
    logger.info("\n✓ 모든 Segmentation 모델 설정 테스트 통과")


def test_data_path():
    """데이터 경로 존재 확인"""
    logger.info("\n" + "=" * 60)
    logger.info("6. 데이터 경로 확인")
    logger.info("=" * 60)
    
    data_yaml = r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml"
    
    if Path(data_yaml).exists():
        logger.info(f"✓ 데이터 파일 존재: {data_yaml}")
        
        # data.yaml 내용 확인
        import yaml
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"\n[data.yaml 내용]")
        logger.info(f"  - Path: {data.get('path')}")
        logger.info(f"  - Task: {data.get('task')}")
        logger.info(f"  - Classes: {data.get('nc')}")
        logger.info(f"  - Names: {data.get('names')}")
        
        assert data.get('task') == 'segment', "Task in data.yaml should be 'segment'"
        assert data.get('nc') == 2, "Should have 2 classes"
        
    else:
        logger.warning(f"⚠ 데이터 파일을 찾을 수 없습니다: {data_yaml}")
        logger.warning("  학습을 실행하기 전에 데이터 경로를 확인하세요.")


def main():
    """전체 테스트 실행"""
    logger.info("\n" + "#" * 60)
    logger.info("# Segmentation Configuration Test Suite")
    logger.info("#" * 60)
    
    try:
        test_model_types()
        test_dataset_types()
        test_segmentation_config()
        test_detection_config()
        test_all_seg_models()
        test_data_path()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 모든 테스트 통과!")
        logger.info("=" * 60)
        logger.info("\nSegmentation 학습을 시작할 수 있습니다:")
        logger.info("  python train_model3_seg.py")
        logger.info("\n또는")
        logger.info("  python optimized_training.py --task segment")
        logger.info("=" * 60)
        
        return True
        
    except AssertionError as e:
        logger.error(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


