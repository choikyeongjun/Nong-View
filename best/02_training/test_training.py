"""
Test suite for Optimized Training System
Author: Claude Opus
Date: 2025-10-28
"""

import unittest
import sys
import os
from pathlib import Path
import torch
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimized_training import (
    TrainingConfig, TrainingStrategy, OptimizedModelTrainer,
    AdvancedLearningRateScheduler, LossFunctionOptimizer,
    EnsembleTrainer, create_training_config
)
from configs.best_config import ModelType, DatasetType


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration"""
    
    def test_config_creation(self):
        """Test creating training configuration"""
        config = TrainingConfig(
            model_type=ModelType.YOLO11N,
            dataset_type=DatasetType.GROWTH_TIF,
            epochs=100,
            batch_size=16,
            imgsz=640
        )
        
        self.assertEqual(config.model_type, ModelType.YOLO11N)
        self.assertEqual(config.dataset_type, DatasetType.GROWTH_TIF)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.batch_size, 16)
        
    def test_config_factory(self):
        """Test configuration factory function"""
        config = create_training_config(
            model_type=ModelType.YOLO11S,
            dataset_type=DatasetType.GREENHOUSE_MULTI,
            strategy=TrainingStrategy.PROGRESSIVE
        )
        
        self.assertEqual(config.strategy, TrainingStrategy.PROGRESSIVE)
        self.assertIsInstance(config, TrainingConfig)
        
        # Check dataset-specific adjustments
        self.assertEqual(config.box_loss_weight, 10.0)  # Adjusted for multi-object


class TestLearningRateScheduler(unittest.TestCase):
    """Test learning rate scheduler"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.config = TrainingConfig(
            model_type=ModelType.YOLO11N,
            dataset_type=DatasetType.GROWTH_TIF,
            epochs=10,
            batch_size=16,
            imgsz=640,
            base_lr=0.01,
            final_lr=0.0001
        )
        
        # Create mock optimizer
        self.optimizer = Mock()
        self.optimizer.param_groups = [{'lr': self.config.base_lr}]
        
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        scheduler = AdvancedLearningRateScheduler(self.optimizer, self.config)
        
        self.assertEqual(scheduler.current_lr, self.config.base_lr)
        self.assertEqual(scheduler.current_epoch, 0)
        
    def test_plateau_detection(self):
        """Test learning rate plateau detection"""
        scheduler = AdvancedLearningRateScheduler(self.optimizer, self.config)
        
        # Simulate plateau with similar losses
        for _ in range(10):
            is_plateau = scheduler._is_plateau(0.5)
        
        self.assertTrue(is_plateau)
        
    def test_lr_reduction_on_plateau(self):
        """Test learning rate reduction on plateau"""
        scheduler = AdvancedLearningRateScheduler(self.optimizer, self.config)
        initial_lr = self.config.base_lr
        
        scheduler._reduce_lr_on_plateau(factor=0.5)
        
        # Check if LR was reduced
        self.assertEqual(self.optimizer.param_groups[0]['lr'], initial_lr * 0.5)


class TestLossFunctionOptimizer(unittest.TestCase):
    """Test loss function optimizer"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.config = TrainingConfig(
            model_type=ModelType.YOLO11N,
            dataset_type=DatasetType.GROWTH_TIF,
            epochs=10,
            batch_size=16,
            imgsz=640
        )
        
    def test_adaptive_weights(self):
        """Test adaptive weight calculation"""
        loss_optimizer = LossFunctionOptimizer(self.config)
        
        weights = loss_optimizer.adaptive_weights
        
        # Check weights exist
        self.assertIn('box', weights)
        self.assertIn('cls', weights)
        self.assertIn('dfl', weights)
        
        # For GROWTH_TIF, classification weight should be increased
        self.assertGreater(weights['cls'], self.config.cls_loss_weight)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_focal_loss(self):
        """Test focal loss computation"""
        loss_optimizer = LossFunctionOptimizer(self.config)
        
        # Create mock tensors
        batch_size = 4
        num_classes = 3
        pred = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))
        
        loss = loss_optimizer.compute_focal_loss(pred, target)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss


class TestOptimizedModelTrainer(unittest.TestCase):
    """Test main trainer class"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.config = TrainingConfig(
            model_type=ModelType.YOLO11N,
            dataset_type=DatasetType.GROWTH_TIF,
            epochs=2,  # Short for testing
            batch_size=4,
            imgsz=320,  # Small for testing
            device='cpu'  # Force CPU for testing
        )
        
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Cleanup after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    @patch('optimized_training.YOLO')
    def test_trainer_initialization(self, mock_yolo):
        """Test trainer initialization"""
        trainer = OptimizedModelTrainer(self.config)
        
        self.assertEqual(trainer.config, self.config)
        self.assertIsNotNone(trainer.loss_optimizer)
        self.assertTrue(trainer.output_dir.exists())
        
    def test_progressive_resizing(self):
        """Test progressive resizing strategy"""
        config = TrainingConfig(
            model_type=ModelType.YOLO11N,
            dataset_type=DatasetType.GROWTH_TIF,
            epochs=100,
            batch_size=16,
            imgsz=640,
            strategy=TrainingStrategy.PROGRESSIVE
        )
        
        trainer = OptimizedModelTrainer(config)
        
        # Test different epoch stages
        self.assertEqual(trainer._apply_progressive_resizing(10), 320)  # Early
        self.assertEqual(trainer._apply_progressive_resizing(30), 416)  # Mid-early
        self.assertEqual(trainer._apply_progressive_resizing(60), 512)  # Mid-late
        self.assertEqual(trainer._apply_progressive_resizing(90), 640)  # Late
        
    def test_curriculum_learning(self):
        """Test curriculum learning strategy"""
        config = TrainingConfig(
            model_type=ModelType.YOLO11N,
            dataset_type=DatasetType.GROWTH_TIF,
            epochs=100,
            batch_size=16,
            imgsz=640,
            strategy=TrainingStrategy.CURRICULUM
        )
        
        trainer = OptimizedModelTrainer(config)
        
        # Test early epoch (easy samples)
        early_params = trainer._apply_curriculum_learning(10)
        self.assertGreater(early_params['confidence_threshold'], 0.5)
        
        # Test late epoch (all samples)
        late_params = trainer._apply_curriculum_learning(90)
        self.assertLess(late_params['confidence_threshold'], 0.4)
        
    @patch('optimized_training.YOLO')
    def test_hardware_optimization(self, mock_yolo):
        """Test hardware optimization setup"""
        if torch.cuda.is_available():
            config = TrainingConfig(
                model_type=ModelType.YOLO11N,
                dataset_type=DatasetType.GROWTH_TIF,
                epochs=2,
                batch_size=4,
                imgsz=320,
                device='cuda'
            )
            
            trainer = OptimizedModelTrainer(config)
            
            # Check if TF32 is enabled
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)


class TestEnsembleTrainer(unittest.TestCase):
    """Test ensemble trainer"""
    
    def test_ensemble_creation(self):
        """Test creating ensemble trainer"""
        configs = [
            create_training_config(ModelType.YOLO11N, DatasetType.GROWTH_TIF),
            create_training_config(ModelType.YOLO11S, DatasetType.GROWTH_TIF),
            create_training_config(ModelType.YOLO11M, DatasetType.GROWTH_TIF)
        ]
        
        ensemble = EnsembleTrainer(configs)
        
        self.assertEqual(len(ensemble.trainers), 3)
        self.assertEqual(len(ensemble.configs), 3)
        
    def test_ensemble_report_generation(self):
        """Test ensemble report generation"""
        configs = [
            create_training_config(ModelType.YOLO11N, DatasetType.GROWTH_TIF),
            create_training_config(ModelType.YOLO11S, DatasetType.GROWTH_TIF)
        ]
        
        ensemble = EnsembleTrainer(configs)
        
        # Mock results
        ensemble.results = [
            {
                'final_metrics': {
                    'mAP50': 0.85,
                    'mAP50-95': 0.65,
                    'precision': 0.88,
                    'recall': 0.82
                }
            },
            {
                'final_metrics': {
                    'mAP50': 0.87,
                    'mAP50-95': 0.67,
                    'precision': 0.89,
                    'recall': 0.84
                }
            }
        ]
        
        report = ensemble._generate_ensemble_report()
        
        self.assertEqual(report['ensemble_size'], 2)
        self.assertAlmostEqual(report['average_metrics']['mAP50'], 0.86, places=2)
        self.assertEqual(report['best_model']['final_metrics']['mAP50'], 0.87)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_memory_management(self):
        """Test CUDA memory management"""
        config = TrainingConfig(
            model_type=ModelType.YOLO11N,
            dataset_type=DatasetType.GROWTH_TIF,
            epochs=1,
            batch_size=2,
            imgsz=320,
            device='cuda'
        )
        
        trainer = OptimizedModelTrainer(config)
        
        # Check memory is properly managed
        initial_memory = torch.cuda.memory_reserved()
        trainer._cleanup()
        final_memory = torch.cuda.memory_reserved()
        
        # Memory should be freed after cleanup
        self.assertLessEqual(final_memory, initial_memory)
        
    def test_training_strategies(self):
        """Test different training strategies"""
        strategies = [
            TrainingStrategy.STANDARD,
            TrainingStrategy.PROGRESSIVE,
            TrainingStrategy.CURRICULUM
        ]
        
        for strategy in strategies:
            config = create_training_config(
                model_type=ModelType.YOLO11N,
                dataset_type=DatasetType.GROWTH_TIF,
                strategy=strategy
            )
            
            self.assertEqual(config.strategy, strategy)
            self.assertIsInstance(config, TrainingConfig)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)