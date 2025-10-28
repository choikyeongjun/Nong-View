"""
Test suite for Performance Benchmark System
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
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_benchmark import (
    BenchmarkConfig, BenchmarkMetrics, PerformanceBenchmark,
    ComparativeBenchmark
)


class TestBenchmarkConfig(unittest.TestCase):
    """Test benchmark configuration"""
    
    def test_config_creation(self):
        """Test creating benchmark configuration"""
        config = BenchmarkConfig(
            model_path="test_model.pt",
            test_data="test_data.yaml",
            num_iterations=50
        )
        
        self.assertEqual(config.model_path, "test_model.pt")
        self.assertEqual(config.test_data, "test_data.yaml")
        self.assertEqual(config.num_iterations, 50)
        
    def test_config_defaults(self):
        """Test configuration default values"""
        config = BenchmarkConfig(
            model_path="test_model.pt",
            test_data="test_data.yaml"
        )
        
        # Check defaults
        self.assertEqual(config.num_iterations, 100)
        self.assertEqual(config.warmup_iterations, 10)
        self.assertIsNotNone(config.batch_sizes)
        self.assertIsNotNone(config.image_sizes)
        
    def test_post_init_defaults(self):
        """Test post-init default assignments"""
        config = BenchmarkConfig(
            model_path="test_model.pt",
            test_data="test_data.yaml",
            batch_sizes=None,
            image_sizes=None
        )
        
        # Should be set in __post_init__
        self.assertEqual(config.batch_sizes, [1, 4, 8, 16, 32])
        self.assertEqual(config.image_sizes, [320, 416, 512, 640, 736])


class TestBenchmarkMetrics(unittest.TestCase):
    """Test benchmark metrics dataclass"""
    
    def test_metrics_creation(self):
        """Test creating benchmark metrics"""
        metrics = BenchmarkMetrics(
            mAP50=0.85,
            mAP50_95=0.65,
            precision=0.88,
            recall=0.82,
            f1_score=0.85,
            inference_time_avg=15.5,
            inference_time_std=2.1,
            preprocessing_time=2.0,
            postprocessing_time=1.5,
            fps=64.5,
            gpu_memory_peak=4.5,
            gpu_memory_avg=3.8,
            cpu_memory_peak=8.2,
            model_size=25.5,
            gflops=15.3,
            params_millions=11.1,
            latency_percentiles={'p50': 14.0, 'p90': 18.0, 'p95': 20.0, 'p99': 25.0},
            gpu_utilization_avg=85.0,
            cpu_utilization_avg=45.0,
            failure_rate=0.0,
            consistency_score=0.95
        )
        
        self.assertEqual(metrics.mAP50, 0.85)
        self.assertEqual(metrics.fps, 64.5)
        self.assertIn('p50', metrics.latency_percentiles)
        
    def test_metrics_serialization(self):
        """Test metrics can be serialized to dict"""
        metrics = BenchmarkMetrics(
            mAP50=0.85,
            mAP50_95=0.65,
            precision=0.88,
            recall=0.82,
            f1_score=0.85,
            inference_time_avg=15.5,
            inference_time_std=2.1,
            preprocessing_time=2.0,
            postprocessing_time=1.5,
            fps=64.5,
            gpu_memory_peak=4.5,
            gpu_memory_avg=3.8,
            cpu_memory_peak=8.2,
            model_size=25.5,
            gflops=15.3,
            params_millions=11.1,
            latency_percentiles={'p50': 14.0},
            gpu_utilization_avg=85.0,
            cpu_utilization_avg=45.0,
            failure_rate=0.0,
            consistency_score=0.95
        )
        
        metrics_dict = asdict(metrics)
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['mAP50'], 0.85)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test main benchmark class"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = BenchmarkConfig(
            model_path="test_model.pt",
            test_data="test_data.yaml",
            num_iterations=10,
            batch_sizes=[1, 2],
            image_sizes=[320, 640],
            output_dir=self.temp_dir,
            device='cpu'
        )
        
    def tearDown(self):
        """Cleanup after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        benchmark = PerformanceBenchmark(self.config)
        
        self.assertEqual(benchmark.config, self.config)
        self.assertTrue(benchmark.output_dir.exists())
        self.assertIsNotNone(benchmark.hardware_info)
        
    def test_hardware_info(self):
        """Test hardware information gathering"""
        benchmark = PerformanceBenchmark(self.config)
        hw_info = benchmark.hardware_info
        
        self.assertIn('cpu', hw_info)
        self.assertIn('cores', hw_info['cpu'])
        self.assertIn('ram_gb', hw_info['cpu'])
        
        if torch.cuda.is_available():
            self.assertIn('gpu', hw_info)
            
    @patch('performance_benchmark.YOLO')
    def test_model_loading(self, mock_yolo):
        """Test model loading"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.model = Mock()
        
        benchmark = PerformanceBenchmark(self.config)
        model = benchmark.load_model("test.pt")
        
        self.assertIsNotNone(model)
        mock_yolo.assert_called_once_with("test.pt")
        
    @patch('performance_benchmark.YOLO')
    def test_warmup(self, mock_yolo):
        """Test model warmup"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        benchmark = PerformanceBenchmark(self.config)
        benchmark._warmup_model(mock_model, iterations=5)
        
        # Should call model 5 times for warmup
        self.assertEqual(mock_model.call_count, 5)
        
    def test_inference_speed_measurement(self):
        """Test inference speed measurement"""
        benchmark = PerformanceBenchmark(self.config)
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = Mock()
        
        # Mock timer results
        with patch('performance_benchmark.Timer') as mock_timer:
            mock_measurement = Mock()
            mock_measurement.mean = 0.015  # 15ms
            mock_timer.return_value.timeit.return_value = mock_measurement
            
            results = benchmark.benchmark_inference_speed(mock_model, batch_size=1, image_size=640)
            
            self.assertIn('mean_time_ms', results)
            self.assertIn('fps', results)
            self.assertIn('throughput_img_per_sec', results)
            
    def test_memory_benchmark(self):
        """Test memory efficiency benchmark"""
        benchmark = PerformanceBenchmark(self.config)
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = Mock()
        
        results = benchmark.benchmark_memory_efficiency(
            mock_model,
            batch_sizes=[1, 2]
        )
        
        self.assertIsInstance(results, dict)
        # Should have results for each batch size
        self.assertIn('batch_1', results)
        
    @patch('performance_benchmark.YOLO')
    def test_model_complexity_calculation(self, mock_yolo):
        """Test model complexity calculation"""
        mock_model = Mock()
        mock_pytorch_model = Mock()
        
        # Mock parameters
        param1 = Mock()
        param1.numel.return_value = 1000000
        param1.requires_grad = True
        
        param2 = Mock()
        param2.numel.return_value = 500000
        param2.requires_grad = False
        
        mock_pytorch_model.parameters.return_value = [param1, param2]
        mock_model.model = mock_pytorch_model
        
        benchmark = PerformanceBenchmark(self.config)
        
        with patch('torch.save'), patch('os.path.getsize', return_value=26214400):  # 25MB
            complexity = benchmark.calculate_model_complexity(mock_model)
            
        self.assertIn('total_params_millions', complexity)
        self.assertIn('model_size_mb', complexity)
        self.assertEqual(complexity['total_params_millions'], 1.5)
        self.assertEqual(complexity['model_size_mb'], 25.0)
        
    def test_final_metrics_creation(self):
        """Test creating final metrics from results"""
        benchmark = PerformanceBenchmark(self.config)
        
        aggregated = {
            'mAP50': [0.85],
            'mAP50-95': [0.65],
            'precision': [0.88],
            'recall': [0.82],
            'f1_score': [0.85]
        }
        
        detailed = {
            'b1_s640': {
                'mean_time_ms': 15.5,
                'std_time_ms': 2.1,
                'fps': 64.5,
                'p50_time_ms': 14.0,
                'p90_time_ms': 18.0,
                'p95_time_ms': 20.0,
                'p99_time_ms': 25.0
            },
            'memory': {
                'batch_1': {
                    'peak_memory_gb': 4.5,
                    'memory_allocated_gb': 3.8
                }
            },
            'complexity': {
                'model_size_mb': 25.5,
                'estimated_gflops': 15.3,
                'total_params_millions': 11.1
            }
        }
        
        metrics = benchmark._create_final_metrics(aggregated, detailed)
        
        self.assertIsInstance(metrics, BenchmarkMetrics)
        self.assertEqual(metrics.mAP50, 0.85)
        self.assertEqual(metrics.fps, 64.5)
        self.assertEqual(metrics.model_size, 25.5)
        
    def test_results_saving(self):
        """Test saving benchmark results"""
        benchmark = PerformanceBenchmark(self.config)
        
        metrics = BenchmarkMetrics(
            mAP50=0.85,
            mAP50_95=0.65,
            precision=0.88,
            recall=0.82,
            f1_score=0.85,
            inference_time_avg=15.5,
            inference_time_std=2.1,
            preprocessing_time=2.0,
            postprocessing_time=1.5,
            fps=64.5,
            gpu_memory_peak=4.5,
            gpu_memory_avg=3.8,
            cpu_memory_peak=8.2,
            model_size=25.5,
            gflops=15.3,
            params_millions=11.1,
            latency_percentiles={'p50': 14.0},
            gpu_utilization_avg=85.0,
            cpu_utilization_avg=45.0,
            failure_rate=0.0,
            consistency_score=0.95
        )
        
        benchmark._save_results(metrics)
        
        # Check files were created
        metrics_file = benchmark.output_dir / 'benchmark_metrics.json'
        self.assertTrue(metrics_file.exists())
        
        # Load and verify
        with open(metrics_file, 'r') as f:
            loaded_metrics = json.load(f)
        
        self.assertEqual(loaded_metrics['mAP50'], 0.85)
        
    def test_report_generation(self):
        """Test report generation"""
        benchmark = PerformanceBenchmark(self.config)
        
        metrics = BenchmarkMetrics(
            mAP50=0.85,
            mAP50_95=0.65,
            precision=0.88,
            recall=0.82,
            f1_score=0.85,
            inference_time_avg=15.5,
            inference_time_std=2.1,
            preprocessing_time=2.0,
            postprocessing_time=1.5,
            fps=64.5,
            gpu_memory_peak=4.5,
            gpu_memory_avg=3.8,
            cpu_memory_peak=8.2,
            model_size=25.5,
            gflops=15.3,
            params_millions=11.1,
            latency_percentiles={'p50': 14.0, 'p95': 20.0, 'p99': 25.0},
            gpu_utilization_avg=85.0,
            cpu_utilization_avg=45.0,
            failure_rate=0.0,
            consistency_score=0.95
        )
        
        report_text = benchmark._generate_report(metrics)
        
        self.assertIn("PERFORMANCE BENCHMARK REPORT", report_text)
        self.assertIn("mAP@50", report_text)
        self.assertIn("FPS", report_text)
        self.assertIn("Model Size", report_text)
        
        # Check file was created
        report_file = benchmark.output_dir / 'benchmark_report.txt'
        self.assertTrue(report_file.exists())


class TestComparativeBenchmark(unittest.TestCase):
    """Test comparative benchmark functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Cleanup after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_comparative_initialization(self):
        """Test comparative benchmark initialization"""
        configs = [
            BenchmarkConfig(
                model_path="model1.pt",
                test_data="test.yaml",
                output_dir=self.temp_dir
            ),
            BenchmarkConfig(
                model_path="model2.pt",
                test_data="test.yaml",
                output_dir=self.temp_dir
            )
        ]
        
        comparative = ComparativeBenchmark(configs)
        
        self.assertEqual(len(comparative.configs), 2)
        self.assertEqual(comparative.configs[0].model_path, "model1.pt")


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_operations(self):
        """Test CUDA-specific operations"""
        config = BenchmarkConfig(
            model_path="test.pt",
            test_data="test.yaml",
            device='cuda',
            num_iterations=5
        )
        
        benchmark = PerformanceBenchmark(config)
        
        # Check CUDA device is properly set
        self.assertEqual(benchmark.device.type, 'cuda')
        
        # Check hardware info includes GPU
        self.assertIn('gpu', benchmark.hardware_info)
        
    def test_cpu_fallback(self):
        """Test CPU fallback when CUDA not available"""
        config = BenchmarkConfig(
            model_path="test.pt",
            test_data="test.yaml",
            device='cpu'
        )
        
        benchmark = PerformanceBenchmark(config)
        
        self.assertEqual(benchmark.device.type, 'cpu')


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)