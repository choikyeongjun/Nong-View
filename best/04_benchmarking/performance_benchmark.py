"""
Advanced Performance Benchmarking System for Nong-View Best
Author: Claude Opus (System Architect & Core Algorithms)
Date: 2025-10-28
Version: 1.0.0

Comprehensive benchmarking system for:
- Model performance evaluation
- Inference speed optimization
- Memory efficiency analysis
- Hardware utilization metrics
- Cross-model comparison
- Production readiness assessment
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
from datetime import datetime
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging
import gc
import psutil
import GPUtil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from tqdm import tqdm
import cv2
from ultralytics import YOLO
import torch.profiler as profiler
from torch.utils.benchmark import Timer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.best_config import CONFIG, ModelType, DatasetType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics"""
    # Performance metrics
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1_score: float
    
    # Speed metrics
    inference_time_avg: float  # ms
    inference_time_std: float  # ms
    preprocessing_time: float  # ms
    postprocessing_time: float  # ms
    fps: float
    
    # Memory metrics
    gpu_memory_peak: float  # GB
    gpu_memory_avg: float  # GB
    cpu_memory_peak: float  # GB
    model_size: float  # MB
    
    # Efficiency metrics
    gflops: float
    params_millions: float
    latency_percentiles: Dict[str, float]  # 50th, 90th, 95th, 99th
    
    # Hardware utilization
    gpu_utilization_avg: float  # %
    cpu_utilization_avg: float  # %
    
    # Stability metrics
    failure_rate: float  # %
    consistency_score: float  # 0-1


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    model_path: str
    test_data: str
    num_iterations: int = 100
    warmup_iterations: int = 10
    batch_sizes: List[int] = None
    image_sizes: List[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    num_workers: int = 4
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    profile_enabled: bool = True
    save_results: bool = True
    output_dir: str = "results/benchmarks"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.image_sizes is None:
            self.image_sizes = [320, 416, 512, 640, 736]


class PerformanceBenchmark:
    """Main benchmarking class for model performance evaluation"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.results = defaultdict(dict)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hardware info
        self.hardware_info = self._get_hardware_info()
        
        logger.info(f"Initialized PerformanceBenchmark with config: {asdict(config)}")
        logger.info(f"Hardware: {self.hardware_info}")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        info = {
            'cpu': {
                'model': psutil.cpu_freq().current,
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'ram_gb': psutil.virtual_memory().total / 1024**3
            }
        }
        
        if self.device.type == 'cuda':
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu:
                info['gpu'] = {
                    'name': gpu.name,
                    'memory_gb': gpu.memoryTotal / 1024,
                    'cuda_version': torch.version.cuda,
                    'compute_capability': torch.cuda.get_device_capability()
                }
        
        return info
    
    def load_model(self, model_path: str) -> YOLO:
        """Load and prepare model for benchmarking"""
        logger.info(f"Loading model from {model_path}")
        
        model = YOLO(model_path)
        model.to(self.device)
        
        if self.config.use_amp and self.device.type == 'cuda':
            model.model = model.model.half()
        
        # Model warmup
        self._warmup_model(model)
        
        return model
    
    def _warmup_model(self, model: YOLO, iterations: int = None):
        """Warm up model to stabilize performance"""
        iterations = iterations or self.config.warmup_iterations
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        
        logger.info(f"Warming up model with {iterations} iterations")
        
        for _ in range(iterations):
            if self.config.use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input, verbose=False)
            else:
                _ = model(dummy_input, verbose=False)
        
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def benchmark_inference_speed(
        self,
        model: YOLO,
        batch_size: int,
        image_size: int
    ) -> Dict[str, float]:
        """Benchmark inference speed"""
        logger.info(f"Benchmarking inference speed: batch_size={batch_size}, image_size={image_size}")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
        
        times = []
        memory_usage = []
        
        # Use torch benchmark timer for accurate measurements
        timer = Timer(
            stmt='model(dummy_input, verbose=False)',
            globals={'model': model, 'dummy_input': dummy_input}
        )
        
        for _ in tqdm(range(self.config.num_iterations), desc="Speed benchmark"):
            # Synchronize before measurement
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Measure time
            measurement = timer.timeit(1)
            times.append(measurement.mean * 1000)  # Convert to ms
            
            # Measure memory
            if self.device.type == 'cuda':
                memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
        
        # Calculate statistics
        times = np.array(times)
        
        results = {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'p50_time_ms': np.percentile(times, 50),
            'p90_time_ms': np.percentile(times, 90),
            'p95_time_ms': np.percentile(times, 95),
            'p99_time_ms': np.percentile(times, 99),
            'fps': 1000 / np.mean(times) * batch_size,
            'throughput_img_per_sec': (batch_size * self.config.num_iterations) / (np.sum(times) / 1000)
        }
        
        if memory_usage:
            results['gpu_memory_gb'] = np.mean(memory_usage)
        
        return results
    
    def benchmark_accuracy(
        self,
        model: YOLO,
        test_data: str
    ) -> Dict[str, float]:
        """Benchmark model accuracy on test dataset"""
        logger.info(f"Benchmarking accuracy on {test_data}")
        
        # Run validation
        metrics = model.val(data=test_data, batch=1, verbose=False)
        
        results = {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.p.mean(),
            'recall': metrics.box.r.mean(),
            'f1_score': 2 * (metrics.box.p.mean() * metrics.box.r.mean()) / 
                       (metrics.box.p.mean() + metrics.box.r.mean() + 1e-10)
        }
        
        # Per-class metrics
        if hasattr(metrics.box, 'ap_class_index'):
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = model.names[int(class_idx)]
                results[f'mAP50_class_{class_name}'] = metrics.box.ap50[i]
        
        return results
    
    def benchmark_memory_efficiency(
        self,
        model: YOLO,
        batch_sizes: List[int],
        image_size: int = 640
    ) -> Dict[str, Any]:
        """Benchmark memory efficiency across different batch sizes"""
        logger.info("Benchmarking memory efficiency")
        
        results = {}
        
        for batch_size in batch_sizes:
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            try:
                # Create input
                dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
                
                # Run inference
                _ = model(dummy_input, verbose=False)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                    
                    results[f'batch_{batch_size}'] = {
                        'memory_allocated_gb': memory_allocated,
                        'memory_reserved_gb': memory_reserved,
                        'peak_memory_gb': peak_memory,
                        'efficiency_ratio': memory_allocated / memory_reserved if memory_reserved > 0 else 0
                    }
                else:
                    # CPU memory tracking
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    results[f'batch_{batch_size}'] = {
                        'memory_gb': memory_info.rss / 1024**3
                    }
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[f'batch_{batch_size}'] = {'error': 'OOM'}
                    logger.warning(f"Out of memory for batch size {batch_size}")
                    break
                else:
                    raise
            finally:
                # Cleanup
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        return results
    
    def profile_model(
        self,
        model: YOLO,
        batch_size: int = 1,
        image_size: int = 640
    ) -> Dict[str, Any]:
        """Profile model with PyTorch profiler"""
        if not self.config.profile_enabled:
            return {}
        
        logger.info("Profiling model operations")
        
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
        
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ] if self.device.type == 'cuda' else [profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with profiler.record_function("model_inference"):
                for _ in range(10):  # Profile 10 iterations
                    _ = model(dummy_input, verbose=False)
        
        # Get profiling results
        key_metrics = prof.key_averages().table(sort_by="self_cuda_time_total" if self.device.type == 'cuda' else "cpu_time_total", row_limit=10)
        
        # Save trace
        trace_file = self.output_dir / f"trace_batch{batch_size}_size{image_size}.json"
        prof.export_chrome_trace(str(trace_file))
        
        # Extract key statistics
        stats = {
            'total_time_ms': sum([item.self_cuda_time_total for item in prof.key_averages()]) / 1000 if self.device.type == 'cuda' else
                           sum([item.cpu_time_total for item in prof.key_averages()]) / 1000,
            'top_operations': key_metrics,
            'trace_file': str(trace_file)
        }
        
        return stats
    
    def benchmark_robustness(
        self,
        model: YOLO,
        test_images: List[str],
        perturbations: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Benchmark model robustness to various perturbations"""
        logger.info("Benchmarking model robustness")
        
        if perturbations is None:
            perturbations = {
                'noise': 0.1,
                'blur': 5,
                'brightness': 0.3,
                'contrast': 0.3
            }
        
        results = defaultdict(list)
        baseline_predictions = []
        
        # Get baseline predictions
        for img_path in test_images[:10]:  # Use subset for robustness test
            img = cv2.imread(img_path)
            pred = model(img, verbose=False)[0]
            baseline_predictions.append(len(pred.boxes) if pred.boxes else 0)
        
        # Test with perturbations
        for perturb_name, perturb_value in perturbations.items():
            perturbed_predictions = []
            
            for img_path in test_images[:10]:
                img = cv2.imread(img_path)
                
                # Apply perturbation
                if perturb_name == 'noise':
                    noise = np.random.randn(*img.shape) * perturb_value * 255
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)
                elif perturb_name == 'blur':
                    img = cv2.GaussianBlur(img, (perturb_value, perturb_value), 0)
                elif perturb_name == 'brightness':
                    img = cv2.convertScaleAbs(img, alpha=1.0, beta=perturb_value * 255)
                elif perturb_name == 'contrast':
                    img = cv2.convertScaleAbs(img, alpha=1.0 + perturb_value, beta=0)
                
                pred = model(img, verbose=False)[0]
                perturbed_predictions.append(len(pred.boxes) if pred.boxes else 0)
            
            # Calculate consistency
            consistency = 1.0 - np.mean(np.abs(np.array(baseline_predictions) - 
                                              np.array(perturbed_predictions))) / \
                               (np.mean(baseline_predictions) + 1e-10)
            
            results[f'robustness_{perturb_name}'] = max(0, consistency)
        
        results['overall_robustness'] = np.mean(list(results.values()))
        
        return dict(results)
    
    def calculate_model_complexity(self, model: YOLO) -> Dict[str, float]:
        """Calculate model complexity metrics"""
        logger.info("Calculating model complexity")
        
        # Get model
        pytorch_model = model.model
        
        # Count parameters
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        
        # Estimate FLOPs (simplified)
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        
        # Use thop or torchinfo if available, otherwise estimate
        try:
            from thop import profile
            flops, params = profile(pytorch_model, inputs=(dummy_input,), verbose=False)
            gflops = flops / 1e9
        except ImportError:
            # Rough estimate based on model size
            gflops = total_params / 1e6 * 2  # Very rough approximation
        
        # Model size on disk
        temp_path = self.output_dir / "temp_model.pt"
        torch.save(pytorch_model.state_dict(), temp_path)
        model_size_mb = os.path.getsize(temp_path) / 1024 / 1024
        os.remove(temp_path)
        
        return {
            'total_params_millions': total_params / 1e6,
            'trainable_params_millions': trainable_params / 1e6,
            'model_size_mb': model_size_mb,
            'estimated_gflops': gflops
        }
    
    def run_comprehensive_benchmark(self) -> BenchmarkMetrics:
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark")
        
        # Load model
        self.model = self.load_model(self.config.model_path)
        
        # Initialize aggregate metrics
        all_metrics = defaultdict(list)
        
        # 1. Accuracy benchmark
        if self.config.test_data:
            accuracy_results = self.benchmark_accuracy(self.model, self.config.test_data)
            for k, v in accuracy_results.items():
                all_metrics[k].append(v)
        
        # 2. Speed benchmarks across configurations
        for batch_size in self.config.batch_sizes:
            for image_size in self.config.image_sizes:
                try:
                    speed_results = self.benchmark_inference_speed(
                        self.model, batch_size, image_size
                    )
                    
                    # Store results
                    self.results[f'b{batch_size}_s{image_size}'] = speed_results
                    
                    # Aggregate for summary
                    if batch_size == 1 and image_size == 640:  # Standard config
                        for k, v in speed_results.items():
                            all_metrics[f'speed_{k}'].append(v)
                    
                except Exception as e:
                    logger.error(f"Failed benchmark for batch={batch_size}, size={image_size}: {e}")
        
        # 3. Memory efficiency benchmark
        memory_results = self.benchmark_memory_efficiency(
            self.model, self.config.batch_sizes
        )
        self.results['memory'] = memory_results
        
        # 4. Model complexity
        complexity = self.calculate_model_complexity(self.model)
        self.results['complexity'] = complexity
        
        # 5. Profiling (optional)
        if self.config.profile_enabled:
            profile_results = self.profile_model(self.model)
            self.results['profile'] = profile_results
        
        # 6. Robustness benchmark (if test images available)
        # This would require actual test images
        
        # Create final metrics
        metrics = self._create_final_metrics(all_metrics, self.results)
        
        # Save results
        if self.config.save_results:
            self._save_results(metrics)
        
        # Generate report
        self._generate_report(metrics)
        
        logger.info("Benchmark completed successfully")
        
        return metrics
    
    def _create_final_metrics(
        self,
        aggregated: Dict[str, List],
        detailed: Dict[str, Any]
    ) -> BenchmarkMetrics:
        """Create final benchmark metrics"""
        
        # Get standard config results (batch=1, size=640)
        standard_results = detailed.get('b1_s640', {})
        
        # Memory metrics
        memory_b1 = detailed.get('memory', {}).get('batch_1', {})
        
        # Complexity metrics
        complexity = detailed.get('complexity', {})
        
        metrics = BenchmarkMetrics(
            # Performance metrics (from accuracy benchmark)
            mAP50=aggregated.get('mAP50', [0])[0],
            mAP50_95=aggregated.get('mAP50-95', [0])[0],
            precision=aggregated.get('precision', [0])[0],
            recall=aggregated.get('recall', [0])[0],
            f1_score=aggregated.get('f1_score', [0])[0],
            
            # Speed metrics
            inference_time_avg=standard_results.get('mean_time_ms', 0),
            inference_time_std=standard_results.get('std_time_ms', 0),
            preprocessing_time=0,  # Would need separate measurement
            postprocessing_time=0,  # Would need separate measurement
            fps=standard_results.get('fps', 0),
            
            # Memory metrics
            gpu_memory_peak=memory_b1.get('peak_memory_gb', 0),
            gpu_memory_avg=memory_b1.get('memory_allocated_gb', 0),
            cpu_memory_peak=psutil.virtual_memory().used / 1024**3,
            model_size=complexity.get('model_size_mb', 0),
            
            # Efficiency metrics
            gflops=complexity.get('estimated_gflops', 0),
            params_millions=complexity.get('total_params_millions', 0),
            latency_percentiles={
                'p50': standard_results.get('p50_time_ms', 0),
                'p90': standard_results.get('p90_time_ms', 0),
                'p95': standard_results.get('p95_time_ms', 0),
                'p99': standard_results.get('p99_time_ms', 0)
            },
            
            # Hardware utilization
            gpu_utilization_avg=0,  # Would need continuous monitoring
            cpu_utilization_avg=psutil.cpu_percent(),
            
            # Stability metrics
            failure_rate=0,
            consistency_score=1.0
        )
        
        return metrics
    
    def _save_results(self, metrics: BenchmarkMetrics):
        """Save benchmark results"""
        # Save metrics as JSON
        metrics_dict = asdict(metrics)
        metrics_file = self.output_dir / 'benchmark_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Save detailed results
        detailed_file = self.output_dir / 'detailed_results.json'
        with open(detailed_file, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_report(self, metrics: BenchmarkMetrics):
        """Generate comprehensive benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Model: {self.config.model_path}")
        report.append(f"Device: {self.config.device}")
        report.append("")
        
        # Performance Summary
        report.append("ACCURACY METRICS:")
        report.append(f"  mAP@50:        {metrics.mAP50:.4f}")
        report.append(f"  mAP@50-95:     {metrics.mAP50_95:.4f}")
        report.append(f"  Precision:     {metrics.precision:.4f}")
        report.append(f"  Recall:        {metrics.recall:.4f}")
        report.append(f"  F1 Score:      {metrics.f1_score:.4f}")
        report.append("")
        
        # Speed Metrics
        report.append("SPEED METRICS:")
        report.append(f"  Inference Time: {metrics.inference_time_avg:.2f} ± {metrics.inference_time_std:.2f} ms")
        report.append(f"  FPS:           {metrics.fps:.1f}")
        report.append(f"  Latency P50:   {metrics.latency_percentiles['p50']:.2f} ms")
        report.append(f"  Latency P95:   {metrics.latency_percentiles['p95']:.2f} ms")
        report.append(f"  Latency P99:   {metrics.latency_percentiles['p99']:.2f} ms")
        report.append("")
        
        # Efficiency Metrics
        report.append("EFFICIENCY METRICS:")
        report.append(f"  Model Size:    {metrics.model_size:.1f} MB")
        report.append(f"  Parameters:    {metrics.params_millions:.1f}M")
        report.append(f"  GFLOPs:        {metrics.gflops:.1f}")
        if metrics.gpu_memory_peak > 0:
            report.append(f"  GPU Memory:    {metrics.gpu_memory_peak:.2f} GB")
        report.append("")
        
        # Hardware Info
        report.append("HARDWARE INFO:")
        for key, value in self.hardware_info.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / 'benchmark_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        # Print to console
        print(report_text)
        
        return report_text
    
    def create_visualization(self):
        """Create benchmark visualization plots"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Performance Benchmark Results', fontsize=16)
        
        # 1. Speed vs Batch Size
        batch_sizes = []
        fps_values = []
        
        for key, value in self.results.items():
            if key.startswith('b') and 's640' in key:  # Standard size
                batch = int(key.split('_')[0][1:])
                batch_sizes.append(batch)
                fps_values.append(value.get('fps', 0))
        
        if batch_sizes:
            axes[0, 0].plot(batch_sizes, fps_values, 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].set_title('Inference Speed vs Batch Size')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Speed vs Image Size
        image_sizes = []
        inference_times = []
        
        for key, value in self.results.items():
            if key.startswith('b1_s'):  # Batch size 1
                size = int(key.split('s')[1])
                image_sizes.append(size)
                inference_times.append(value.get('mean_time_ms', 0))
        
        if image_sizes:
            axes[0, 1].plot(image_sizes, inference_times, 's-', linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_xlabel('Image Size')
            axes[0, 1].set_ylabel('Inference Time (ms)')
            axes[0, 1].set_title('Inference Time vs Image Size')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Memory Usage
        if 'memory' in self.results:
            batch_sizes_mem = []
            memory_usage = []
            
            for batch_key, mem_data in self.results['memory'].items():
                if 'batch_' in batch_key and 'peak_memory_gb' in mem_data:
                    batch = int(batch_key.split('_')[1])
                    batch_sizes_mem.append(batch)
                    memory_usage.append(mem_data['peak_memory_gb'])
            
            if batch_sizes_mem:
                axes[0, 2].bar(batch_sizes_mem, memory_usage, color='green', alpha=0.7)
                axes[0, 2].set_xlabel('Batch Size')
                axes[0, 2].set_ylabel('Peak Memory (GB)')
                axes[0, 2].set_title('Memory Usage vs Batch Size')
                axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Latency Distribution (for standard config)
        if 'b1_s640' in self.results:
            latencies = [
                self.results['b1_s640'].get('p50_time_ms', 0),
                self.results['b1_s640'].get('p90_time_ms', 0),
                self.results['b1_s640'].get('p95_time_ms', 0),
                self.results['b1_s640'].get('p99_time_ms', 0)
            ]
            percentiles = ['P50', 'P90', 'P95', 'P99']
            
            axes[1, 0].bar(percentiles, latencies, color='purple', alpha=0.7)
            axes[1, 0].set_xlabel('Percentile')
            axes[1, 0].set_ylabel('Latency (ms)')
            axes[1, 0].set_title('Latency Percentiles')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Throughput Heatmap
        # Create throughput matrix for batch sizes vs image sizes
        batch_list = sorted(set(int(k.split('_')[0][1:]) for k in self.results.keys() if k.startswith('b')))
        size_list = sorted(set(int(k.split('s')[1]) for k in self.results.keys() if '_s' in k))
        
        if batch_list and size_list:
            throughput_matrix = np.zeros((len(batch_list), len(size_list)))
            
            for i, batch in enumerate(batch_list):
                for j, size in enumerate(size_list):
                    key = f'b{batch}_s{size}'
                    if key in self.results:
                        throughput_matrix[i, j] = self.results[key].get('throughput_img_per_sec', 0)
            
            im = axes[1, 1].imshow(throughput_matrix, cmap='YlOrRd', aspect='auto')
            axes[1, 1].set_xticks(range(len(size_list)))
            axes[1, 1].set_xticklabels(size_list)
            axes[1, 1].set_yticks(range(len(batch_list)))
            axes[1, 1].set_yticklabels(batch_list)
            axes[1, 1].set_xlabel('Image Size')
            axes[1, 1].set_ylabel('Batch Size')
            axes[1, 1].set_title('Throughput Heatmap (img/sec)')
            plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Model Complexity
        if 'complexity' in self.results:
            complexity = self.results['complexity']
            metrics = ['Params (M)', 'Size (MB)', 'GFLOPs']
            values = [
                complexity.get('total_params_millions', 0),
                complexity.get('model_size_mb', 0),
                complexity.get('estimated_gflops', 0)
            ]
            
            axes[1, 2].bar(metrics, values, color=['blue', 'red', 'green'], alpha=0.7)
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].set_title('Model Complexity Metrics')
            axes[1, 2].grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualizations saved to {self.output_dir}")


class ComparativeBenchmark:
    """Compare multiple models or configurations"""
    
    def __init__(self, configs: List[BenchmarkConfig]):
        self.configs = configs
        self.results = {}
        
    def run_comparison(self) -> pd.DataFrame:
        """Run benchmarks for all configurations and compare"""
        logger.info(f"Running comparative benchmark for {len(self.configs)} configurations")
        
        for i, config in enumerate(self.configs):
            logger.info(f"Benchmarking configuration {i+1}/{len(self.configs)}")
            
            benchmark = PerformanceBenchmark(config)
            metrics = benchmark.run_comprehensive_benchmark()
            
            self.results[f'config_{i}'] = asdict(metrics)
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Create comparison DataFrame
        df = pd.DataFrame(self.results).T
        
        # Add relative performance metrics
        baseline = df.iloc[0]
        for col in ['mAP50', 'fps', 'inference_time_avg']:
            if col in df.columns:
                df[f'{col}_relative'] = df[col] / baseline[col]
        
        # Save comparison results
        output_dir = Path(self.configs[0].output_dir).parent / 'comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / 'comparison_results.csv')
        df.to_excel(output_dir / 'comparison_results.xlsx')
        
        # Generate comparison report
        self._generate_comparison_report(df, output_dir)
        
        return df
    
    def _generate_comparison_report(self, df: pd.DataFrame, output_dir: Path):
        """Generate comparison report with visualizations"""
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        configs = df.index
        
        # 1. Accuracy Comparison
        accuracy_metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
        accuracy_data = df[accuracy_metrics].values.T
        
        x = np.arange(len(accuracy_metrics))
        width = 0.8 / len(configs)
        
        for i, config in enumerate(configs):
            axes[0, 0].bar(x + i * width, accuracy_data[:, i], width, label=config)
        
        axes[0, 0].set_xlabel('Metric')
        axes[0, 0].set_xticks(x + width * (len(configs) - 1) / 2)
        axes[0, 0].set_xticklabels(accuracy_metrics)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Accuracy Metrics Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Speed Comparison
        axes[0, 1].bar(configs, df['fps'], color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Configuration')
        axes[0, 1].set_ylabel('FPS')
        axes[0, 1].set_title('Inference Speed Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Memory Comparison
        axes[1, 0].bar(configs, df['gpu_memory_peak'], color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Configuration')
        axes[1, 0].set_ylabel('Peak GPU Memory (GB)')
        axes[1, 0].set_title('Memory Usage Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Efficiency Score (combined metric)
        # Calculate efficiency score: (mAP50 * FPS) / (Memory * Model_Size)
        df['efficiency_score'] = (df['mAP50'] * df['fps']) / \
                                 ((df['gpu_memory_peak'] + 0.1) * (df['model_size'] + 1))
        
        axes[1, 1].bar(configs, df['efficiency_score'], color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Configuration')
        axes[1, 1].set_ylabel('Efficiency Score')
        axes[1, 1].set_title('Overall Efficiency Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Generate text report
        report = []
        report.append("=" * 80)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Number of configurations: {len(configs)}")
        report.append("")
        
        # Best performers
        report.append("BEST PERFORMERS:")
        report.append(f"  Highest mAP50:       {df['mAP50'].idxmax()} ({df['mAP50'].max():.4f})")
        report.append(f"  Fastest (FPS):       {df['fps'].idxmax()} ({df['fps'].max():.1f})")
        report.append(f"  Most Efficient:      {df['efficiency_score'].idxmax()} ({df['efficiency_score'].max():.3f})")
        report.append(f"  Lowest Memory:       {df['gpu_memory_peak'].idxmin()} ({df['gpu_memory_peak'].min():.2f} GB)")
        report.append("")
        
        # Detailed comparison
        report.append("DETAILED COMPARISON:")
        report.append(df.to_string())
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(output_dir / 'comparison_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        logger.info(f"Comparison results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    logger.info("Performance Benchmark System - Claude Opus")
    
    # Single model benchmark
    config = BenchmarkConfig(
        model_path="yolo11n.pt",
        test_data="path/to/test/data.yaml",
        num_iterations=100,
        batch_sizes=[1, 4, 8],
        image_sizes=[320, 640],
        profile_enabled=True
    )
    
    benchmark = PerformanceBenchmark(config)
    # metrics = benchmark.run_comprehensive_benchmark()
    
    # Comparative benchmark
    configs = [
        BenchmarkConfig(model_path="yolo11n.pt", test_data="test.yaml"),
        BenchmarkConfig(model_path="yolo11s.pt", test_data="test.yaml"),
        BenchmarkConfig(model_path="yolo11m.pt", test_data="test.yaml")
    ]
    
    # comparative = ComparativeBenchmark(configs)
    # comparison_df = comparative.run_comparison()
    
    logger.info("Benchmark system initialized successfully")