"""
Model3 Greenhouse Segmentation Training Script
데이터: model3_greenhouse_seg_processed
모델: YOLOv11-seg
태스크: Segmentation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.best_config import ModelType, DatasetType
from optimized_training import (
    OptimizedModelTrainer,
    create_training_config,
    TrainingStrategy
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Model3 Greenhouse Segmentation 학습 실행"""
    
    logger.info("=" * 80)
    logger.info("Model3 Greenhouse Segmentation Training")
    logger.info("=" * 80)
    logger.info("데이터셋: model3_greenhouse_seg_processed")
    logger.info("클래스: Greenhouse_single (단동), Greenhouse_multi (연동)")
    logger.info("모델: YOLOv11-seg")
    logger.info("=" * 80)
    
    # 데이터 경로
    data_yaml = r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml"
    
    # 데이터 파일 존재 확인
    if not Path(data_yaml).exists():
        logger.error(f"데이터 파일을 찾을 수 없습니다: {data_yaml}")
        return
    
    logger.info(f"데이터 파일 확인: {data_yaml}")
    
    # 학습 설정 생성
    config = create_training_config(
        model_type=ModelType.YOLO11N_SEG,  # nano, small, medium, large, xlarge 선택 가능
        dataset_type=DatasetType.MODEL3_GREENHOUSE_SEG,
        strategy=TrainingStrategy.PROGRESSIVE
    )
    
    # 설정 조정 (필요시)
    config.epochs = 100
    config.batch_size = 16
    config.imgsz = 640
    config.patience = 30
    
    logger.info("\n학습 설정:")
    logger.info(f"  - 모델: {config.model_type.name}")
    logger.info(f"  - 에포크: {config.epochs}")
    logger.info(f"  - 배치 크기: {config.batch_size}")
    logger.info(f"  - 이미지 크기: {config.imgsz}")
    logger.info(f"  - 옵티마이저: {config.optimizer}")
    logger.info(f"  - 학습률: {config.base_lr}")
    logger.info(f"  - 전략: {config.strategy.value}")
    logger.info(f"  - Task: {config.task}")
    logger.info(f"  - Overlap Mask: {config.overlap_mask}")
    logger.info(f"  - Mask Ratio: {config.mask_ratio}")
    
    logger.info("\n손실 가중치:")
    logger.info(f"  - Box Loss: {config.box_loss_weight}")
    logger.info(f"  - Cls Loss: {config.cls_loss_weight}")
    logger.info(f"  - DFL Loss: {config.dfl_loss_weight}")
    logger.info(f"  - Mask Loss: {config.mask_loss_weight}")
    
    # 학습 시작
    try:
        logger.info("\n" + "=" * 80)
        logger.info("학습을 시작합니다...")
        logger.info("=" * 80)
        
        trainer = OptimizedModelTrainer(config)
        results = trainer.train(data_yaml)
        
        # 결과 출력
        logger.info("\n" + "=" * 80)
        logger.info("🎉 학습 완료!")
        logger.info("=" * 80)
        logger.info("\n최고 성능:")
        logger.info(f"  - Box mAP50: {results['best_metrics']['mAP50']:.4f}")
        logger.info(f"  - Box mAP50-95: {results['best_metrics']['mAP50-95']:.4f}")
        logger.info(f"  - Mask mAP50: {results['best_metrics'].get('mask_mAP50', 0):.4f}")
        logger.info(f"  - Mask mAP50-95: {results['best_metrics'].get('mask_mAP50-95', 0):.4f}")
        
        logger.info("\n최종 메트릭:")
        for key, value in results['final_metrics'].items():
            logger.info(f"  - {key}: {value:.4f}")
        
        logger.info(f"\n학습 시간: {results['training_time']/3600:.2f}시간")
        logger.info(f"출력 디렉토리: {trainer.output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

