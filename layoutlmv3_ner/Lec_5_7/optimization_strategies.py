import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import psutil
import gc
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.ensemble import VotingClassifier
from collections import Counter, defaultdict
import json
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class OptimizationConfig:
    """优化配置"""
    # 图像优化参数
    denoise_strength: float = 10.0
    sharpen_strength: float = 1.0
    contrast_factor: float = 1.2
    brightness_factor: float = 1.1
    
    # 模型优化参数
    batch_size: int = 8
    max_workers: int = 4
    use_gpu: bool = True
    mixed_precision: bool = True
    
    # 缓存参数
    enable_cache: bool = True
    cache_size: int = 1000
    
    # 集成参数
    ensemble_method: str = "voting"  # voting, stacking, boosting
    confidence_threshold: float = 0.8

class PerformanceOptimizer:
    """性能优化策略"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 缓存系统
        self.image_cache = {} if self.config.enable_cache else None
        self.prediction_cache = {} if self.config.enable_cache else None
        
        # 性能统计
        self.performance_stats = {
            'image_processing_time': [],
            'model_inference_time': [],
            'total_processing_time': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # GPU设置
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu')
        
    def optimize_ocr_quality(self, image_path: str, advanced: bool = True) -> str:
        """OCR质量优化"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"{image_path}_{advanced}"
        if self.image_cache and cache_key in self.image_cache:
            self.performance_stats['cache_hits'] += 1
            return self.image_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image from {image_path}")
        
        if advanced:
            optimized_image = self._advanced_image_preprocessing(image)
        else:
            optimized_image = self._basic_image_preprocessing(image)
        
        # 保存优化后的图像
        base_name = Path(image_path).stem
        suffix = "_advanced" if advanced else "_basic"
        optimized_path = str(Path(image_path).parent / f"{base_name}{suffix}_optimized.jpg")
        cv2.imwrite(optimized_path, optimized_image)
        
        # 缓存结果
        if self.image_cache:
            if len(self.image_cache) >= self.config.cache_size:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            self.image_cache[cache_key] = optimized_path
        
        processing_time = time.time() - start_time
        self.performance_stats['image_processing_time'].append(processing_time)
        
        return optimized_path
    
    def _basic_image_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """基础图像预处理"""
        # 1. 去噪
        denoised = cv2.fastNlMeansDenoising(image, h=self.config.denoise_strength)
        
        # 2. 锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 3. 二值化
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _advanced_image_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """高级图像预处理"""
        # 转换为PIL图像以使用更多预处理选项
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. 对比度增强
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(self.config.contrast_factor)
        
        # 2. 亮度调整
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(self.config.brightness_factor)
        
        # 3. 锐化
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        # 转回OpenCV格式
        cv_image = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        
        # 4. 高级去噪
        denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)
        
        # 5. 形态学操作
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 6. 自适应二值化
        binary = cv2.adaptiveThreshold(
            morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def optimize_model_inference(self, model, data_loader: DataLoader) -> List[Dict]:
        """模型推理优化"""
        start_time = time.time()
        
        model.to(self.device)
        model.eval()
        
        results = []
        
        # 使用混合精度训练
        if self.config.mixed_precision and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                results = self._batch_inference(model, data_loader)
        else:
            results = self._batch_inference(model, data_loader)
        
        inference_time = time.time() - start_time
        self.performance_stats['model_inference_time'].append(inference_time)
        
        return results
    
    def _batch_inference(self, model, data_loader: DataLoader) -> List[Dict]:
        """批量推理"""
        results = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 推理
                outputs = model(**batch)
                
                # 处理输出
                batch_results = self._process_model_outputs(outputs, batch)
                results.extend(batch_results)
                
                # 清理GPU内存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return results
    
    def _process_model_outputs(self, outputs, batch) -> List[Dict]:
        """处理模型输出"""
        # 这里需要根据具体模型输出格式进行调整
        predictions = torch.argmax(outputs.logits, dim=-1)
        confidences = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0]
        
        batch_results = []
        for i in range(len(predictions)):
            result = {
                'predictions': predictions[i].cpu().numpy().tolist(),
                'confidences': confidences[i].cpu().numpy().tolist()
            }
            batch_results.append(result)
        
        return batch_results
    
    def ensemble_prediction(self, models: List, input_data: Dict, method: str = None) -> Dict:
        """集成预测"""
        method = method or self.config.ensemble_method
        
        # 检查缓存
        cache_key = f"{hash(str(input_data))}_{method}"
        if self.prediction_cache and cache_key in self.prediction_cache:
            self.performance_stats['cache_hits'] += 1
            return self.prediction_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        
        if method == "voting":
            result = self._voting_ensemble(models, input_data)
        elif method == "stacking":
            result = self._stacking_ensemble(models, input_data)
        elif method == "boosting":
            result = self._boosting_ensemble(models, input_data)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # 缓存结果
        if self.prediction_cache:
            if len(self.prediction_cache) >= self.config.cache_size:
                oldest_key = next(iter(self.prediction_cache))
                del self.prediction_cache[oldest_key]
            self.prediction_cache[cache_key] = result
        
        return result
    
    def _voting_ensemble(self, models: List, input_data: Dict) -> Dict:
        """投票集成"""
        predictions = []
        confidences = []
        
        for model in models:
            pred = model.predict(input_data)
            predictions.append(pred['prediction'])
            confidences.append(pred.get('confidence', 0.5))
        
        # 加权投票
        final_prediction = self._weighted_vote(predictions, confidences)
        avg_confidence = np.mean(confidences)
        
        return {
            'prediction': final_prediction,
            'confidence': avg_confidence,
            'individual_predictions': predictions,
            'individual_confidences': confidences
        }
    
    def _weighted_vote(self, predictions: List, confidences: List) -> Dict:
        """加权投票"""
        # 按字段进行投票
        field_votes = defaultdict(list)
        
        for pred, conf in zip(predictions, confidences):
            for field, value in pred.items():
                field_votes[field].append((value, conf))
        
        final_prediction = {}
        for field, votes in field_votes.items():
            # 计算加权投票结果
            value_weights = defaultdict(float)
            for value, weight in votes:
                value_weights[value] += weight
            
            # 选择权重最高的值
            best_value = max(value_weights.items(), key=lambda x: x[1])[0]
            final_prediction[field] = best_value
        
        return final_prediction
    
    def _stacking_ensemble(self, models: List, input_data: Dict) -> Dict:
        """堆叠集成"""
        # 获取基模型预测
        base_predictions = []
        for model in models:
            pred = model.predict(input_data)
            base_predictions.append(pred)
        
        # 这里应该有一个元学习器来组合基模型预测
        # 简化实现：使用平均值
        final_prediction = self._average_predictions(base_predictions)
        
        return final_prediction
    
    def _boosting_ensemble(self, models: List, input_data: Dict) -> Dict:
        """提升集成"""
        # 简化的boosting实现
        # 根据模型性能给予不同权重
        weights = [0.4, 0.3, 0.2, 0.1][:len(models)]  # 假设权重
        
        predictions = []
        for model, weight in zip(models, weights):
            pred = model.predict(input_data)
            pred['weight'] = weight
            predictions.append(pred)
        
        final_prediction = self._weighted_average_predictions(predictions)
        
        return final_prediction
    
    def _average_predictions(self, predictions: List[Dict]) -> Dict:
        """平均预测结果"""
        if not predictions:
            return {}
        
        # 收集所有字段
        all_fields = set()
        for pred in predictions:
            all_fields.update(pred.get('prediction', {}).keys())
        
        final_prediction = {}
        for field in all_fields:
            values = []
            for pred in predictions:
                value = pred.get('prediction', {}).get(field)
                if value is not None:
                    values.append(value)
            
            if values:
                # 对于数值字段，计算平均值；对于文本字段，使用众数
                if all(isinstance(v, (int, float)) for v in values):
                    final_prediction[field] = np.mean(values)
                else:
                    final_prediction[field] = Counter(values).most_common(1)[0][0]
        
        return {'prediction': final_prediction}
    
    def _weighted_average_predictions(self, predictions: List[Dict]) -> Dict:
        """加权平均预测结果"""
        if not predictions:
            return {}
        
        all_fields = set()
        for pred in predictions:
            all_fields.update(pred.get('prediction', {}).keys())
        
        final_prediction = {}
        for field in all_fields:
            weighted_values = []
            total_weight = 0
            
            for pred in predictions:
                value = pred.get('prediction', {}).get(field)
                weight = pred.get('weight', 1.0)
                
                if value is not None:
                    weighted_values.append((value, weight))
                    total_weight += weight
            
            if weighted_values and total_weight > 0:
                if all(isinstance(v, (int, float)) for v, w in weighted_values):
                    # 数值字段：加权平均
                    weighted_sum = sum(v * w for v, w in weighted_values)
                    final_prediction[field] = weighted_sum / total_weight
                else:
                    # 文本字段：加权投票
                    value_weights = defaultdict(float)
                    for value, weight in weighted_values:
                        value_weights[value] += weight
                    final_prediction[field] = max(value_weights.items(), key=lambda x: x[1])[0]
        
        return {'prediction': final_prediction}
    
    def parallel_processing(self, processing_func, data_list: List, max_workers: int = None) -> List:
        """并行处理"""
        max_workers = max_workers or self.config.max_workers
        
        # 根据任务类型选择执行器
        if self._is_cpu_intensive(processing_func):
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        results = []
        with executor_class(max_workers=max_workers) as executor:
            future_to_data = {executor.submit(processing_func, data): data for data in data_list}
            
            for future in future_to_data:
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Processing failed for data {future_to_data[future]}: {e}")
                    results.append(None)
        
        return results
    
    def _is_cpu_intensive(self, func) -> bool:
        """判断是否为CPU密集型任务"""
        # 简单的启发式判断
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        cpu_intensive_keywords = ['process', 'compute', 'calculate', 'transform']
        return any(keyword in func_name.lower() for keyword in cpu_intensive_keywords)
    
    def memory_optimization(self):
        """内存优化"""
        # 清理缓存
        if self.image_cache:
            cache_size = len(self.image_cache)
            self.image_cache.clear()
            self.logger.info(f"Cleared image cache: {cache_size} items")
        
        if self.prediction_cache:
            cache_size = len(self.prediction_cache)
            self.prediction_cache.clear()
            self.logger.info(f"Cleared prediction cache: {cache_size} items")
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 记录内存使用情况
        memory_info = psutil.virtual_memory()
        self.logger.info(f"Memory usage after optimization: {memory_info.percent:.1f}%")
    
    def adaptive_batch_size(self, model, sample_data, target_memory_usage: float = 0.8) -> int:
        """自适应批量大小"""
        if not torch.cuda.is_available():
            return self.config.batch_size
        
        # 测试不同批量大小的内存使用
        test_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        optimal_batch_size = 1
        
        model.eval()
        with torch.no_grad():
            for batch_size in test_batch_sizes:
                try:
                    # 创建测试批次
                    test_batch = self._create_test_batch(sample_data, batch_size)
                    
                    # 清理GPU内存
                    torch.cuda.empty_cache()
                    
                    # 记录初始内存
                    initial_memory = torch.cuda.memory_allocated()
                    
                    # 前向传播
                    _ = model(**test_batch)
                    
                    # 记录峰值内存
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_usage = peak_memory / torch.cuda.get_device_properties(0).total_memory
                    
                    if memory_usage <= target_memory_usage:
                        optimal_batch_size = batch_size
                    else:
                        break
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    else:
                        raise e
        
        self.logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size
    
    def _create_test_batch(self, sample_data: Dict, batch_size: int) -> Dict:
        """创建测试批次"""
        test_batch = {}
        for key, value in sample_data.items():
            if isinstance(value, torch.Tensor):
                # 复制张量到指定批量大小
                repeated_tensor = value.unsqueeze(0).repeat(batch_size, *([1] * len(value.shape)))
                test_batch[key] = repeated_tensor.to(self.device)
            else:
                test_batch[key] = [value] * batch_size
        
        return test_batch
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'cache_statistics': {
                'hits': self.performance_stats['cache_hits'],
                'misses': self.performance_stats['cache_misses'],
                'hit_rate': (
                    self.performance_stats['cache_hits'] / 
                    (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
                    if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0 else 0
                )
            },
            'timing_statistics': {
                'avg_image_processing_time': np.mean(self.performance_stats['image_processing_time']) if self.performance_stats['image_processing_time'] else 0,
                'avg_model_inference_time': np.mean(self.performance_stats['model_inference_time']) if self.performance_stats['model_inference_time'] else 0,
                'total_requests': len(self.performance_stats['image_processing_time'])
            },
            'system_resources': {
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(),
                'gpu_available': torch.cuda.is_available(),
                'gpu_memory_usage': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100 if torch.cuda.is_available() else 0
            }
        }
        
        return report
    
    def save_optimization_config(self, config_path: str):
        """保存优化配置"""
        config_dict = {
            'denoise_strength': self.config.denoise_strength,
            'sharpen_strength': self.config.sharpen_strength,
            'contrast_factor': self.config.contrast_factor,
            'brightness_factor': self.config.brightness_factor,
            'batch_size': self.config.batch_size,
            'max_workers': self.config.max_workers,
            'use_gpu': self.config.use_gpu,
            'mixed_precision': self.config.mixed_precision,
            'enable_cache': self.config.enable_cache,
            'cache_size': self.config.cache_size,
            'ensemble_method': self.config.ensemble_method,
            'confidence_threshold': self.config.confidence_threshold
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Optimization config saved to {config_path}")
    
    def load_optimization_config(self, config_path: str):
        """加载优化配置"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.logger.info(f"Optimization config loaded from {config_path}")