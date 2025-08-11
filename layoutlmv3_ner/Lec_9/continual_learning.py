import torch
import torch.nn as nn
import numpy as np
from collections import deque, defaultdict
import random
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
import json
import logging
from typing import Dict, List, Tuple, Any

class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size=10000, importance_sampling=True):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.importance_weights = deque(maxlen=max_size)
        self.importance_sampling = importance_sampling
        
    def add(self, batch, importance_weight=1.0):
        """添加样本到缓冲区"""
        self.buffer.append(batch)
        self.importance_weights.append(importance_weight)
        
    def sample(self, batch_size=32):
        """从缓冲区采样"""
        if len(self.buffer) == 0:
            return None
            
        if self.importance_sampling and len(self.importance_weights) > 0:
            # 重要性采样
            weights = np.array(self.importance_weights)
            weights = weights / weights.sum()
            indices = np.random.choice(
                len(self.buffer), 
                size=min(batch_size, len(self.buffer)), 
                p=weights, 
                replace=False
            )
            return [self.buffer[i] for i in indices]
        else:
            # 随机采样
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class PerformanceMonitor:
    """性能监控器 script"""
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.statistics_history = []
        self.performance_threshold = 0.05  # 性能下降阈值
        
    def update_metrics(self, metrics: Dict[str, float]):
        """更新性能指标"""
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
    def update_statistics(self, stats: Dict[str, Any]):
        """更新统计信息"""
        self.statistics_history.append(stats)
        
    def get_historical_stats(self):
        """获取历史统计信息"""
        if not self.statistics_history:
            return {}
        return self.statistics_history[-1]
        
    def detect_performance_degradation(self) -> bool:
        """检测性能退化"""
        if len(self.metrics_history['accuracy']) < 10:
            return False
            
        recent_acc = np.mean(self.metrics_history['accuracy'][-5:])
        historical_acc = np.mean(self.metrics_history['accuracy'][-15:-5])
        
        return (historical_acc - recent_acc) > self.performance_threshold

class ContinualLearningSystem:
    def __init__(self, model, learning_rate=1e-4, ewc_lambda=1000):
        self.model = model
        self.memory_buffer = ExperienceReplayBuffer()
        self.performance_monitor = PerformanceMonitor()
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda
        self.shift_threshold = 0.1
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # EWC相关
        self.fisher_information = {}
        self.optimal_params = {}
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def online_learning(self, new_data_stream):
        """在线学习"""
        for batch in new_data_stream:
            # 检测数据分布变化
            if self.detect_distribution_shift(batch):
                self.logger.info("Distribution shift detected, adapting model...")
                self.adapt_to_new_distribution(batch)
            
            # 增量学习
            self.incremental_update(batch)
            
            # 经验回放
            if len(self.memory_buffer) > 0:
                replay_batch = self.memory_buffer.sample()
                self.rehearsal_training(replay_batch)
            
            # 存储重要样本
            importance = self.compute_sample_importance(batch)
            self.memory_buffer.add(batch, importance)
            
            # 性能监控
            metrics = self.evaluate_batch(batch)
            self.performance_monitor.update_metrics(metrics)
    
    def detect_distribution_shift(self, batch):
        """检测数据分布变化"""
        current_stats = self.compute_batch_statistics(batch)
        historical_stats = self.performance_monitor.get_historical_stats()
        
        if not historical_stats:
            self.performance_monitor.update_statistics(current_stats)
            return False
        
        # KL散度检测
        kl_divergence = self.compute_kl_divergence(
            current_stats, 
            historical_stats
        )
        
        self.performance_monitor.update_statistics(current_stats)
        return kl_divergence > self.shift_threshold
    
    def compute_batch_statistics(self, batch):
        """计算批次统计信息"""
        # 提取特征统计
        if isinstance(batch, dict) and 'input_ids' in batch:
            # 文本数据
            input_ids = batch['input_ids']
            stats = {
                'mean_length': float(torch.mean(torch.sum(input_ids != 0, dim=1))),
                'vocab_distribution': torch.bincount(input_ids.flatten()).float(),
                'batch_size': input_ids.size(0)
            }
        else:
            # 图像或其他数据
            if torch.is_tensor(batch):
                stats = {
                    'mean': float(torch.mean(batch)),
                    'std': float(torch.std(batch)),
                    'min': float(torch.min(batch)),
                    'max': float(torch.max(batch)),
                    'batch_size': batch.size(0)
                }
            else:
                stats = {'batch_size': len(batch)}
        
        return stats
    
    def compute_kl_divergence(self, current_stats, historical_stats):
        """计算KL散度"""
        try:
            if 'vocab_distribution' in current_stats and 'vocab_distribution' in historical_stats:
                # 文本数据的词汇分布KL散度
                current_dist = current_stats['vocab_distribution']
                historical_dist = historical_stats['vocab_distribution']
                
                # 确保分布长度一致
                max_len = max(len(current_dist), len(historical_dist))
                current_dist = torch.cat([current_dist, torch.zeros(max_len - len(current_dist))])
                historical_dist = torch.cat([historical_dist, torch.zeros(max_len - len(historical_dist))])
                
                # 添加平滑项避免零概率
                current_dist = (current_dist + 1e-8) / (current_dist.sum() + 1e-8 * len(current_dist))
                historical_dist = (historical_dist + 1e-8) / (historical_dist.sum() + 1e-8 * len(historical_dist))
                
                kl_div = torch.sum(current_dist * torch.log(current_dist / historical_dist))
                return float(kl_div)
            else:
                # 数值特征的KL散度近似
                current_mean = current_stats.get('mean', 0)
                historical_mean = historical_stats.get('mean', 0)
                current_std = current_stats.get('std', 1)
                historical_std = historical_stats.get('std', 1)
                
                # 使用正态分布近似计算KL散度
                kl_div = np.log(historical_std / current_std) + \
                        (current_std**2 + (current_mean - historical_mean)**2) / (2 * historical_std**2) - 0.5
                
                return abs(kl_div)
        except Exception as e:
            self.logger.warning(f"KL divergence computation failed: {e}")
            return 0.0
    
    def adapt_to_new_distribution(self, batch):
        """适应新的数据分布"""
        # 调整学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate * 2
        
        # 增加训练轮数
        for _ in range(3):
            self.incremental_update(batch)
        
        # 恢复学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    
    def incremental_update(self, batch):
        """增量更新模型"""
        self.model.train()
        
        # 前向传播
        if isinstance(batch, dict):
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            # 假设是(input, target)格式
            inputs, targets = batch
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # EWC正则化
        if self.fisher_information:
            ewc_loss = self.compute_ewc_loss()
            loss += self.ewc_lambda * ewc_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss)
    
    def rehearsal_training(self, replay_batches):
        """经验回放训练"""
        if not replay_batches:
            return
        
        for batch in replay_batches:
            self.incremental_update(batch)
    
    def compute_sample_importance(self, batch):
        """计算样本重要性"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(batch, dict):
                outputs = self.model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            else:
                inputs, _ = batch
                logits = self.model(inputs)
            
            # 使用预测不确定性作为重要性度量
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            importance = float(torch.mean(entropy))
            
        return importance
    
    def evaluate_batch(self, batch):
        """评估批次性能"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(batch, dict):
                outputs = self.model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                labels = batch.get('labels')
            else:
                inputs, labels = batch
                logits = self.model(inputs)
            
            if labels is not None:
                predictions = torch.argmax(logits, dim=-1)
                accuracy = float(torch.mean((predictions == labels).float()))
                
                # 转换为numpy进行F1计算
                pred_np = predictions.cpu().numpy().flatten()
                label_np = labels.cpu().numpy().flatten()
                f1 = f1_score(label_np, pred_np, average='weighted')
                
                return {'accuracy': accuracy, 'f1_score': f1}
            else:
                return {'accuracy': 0.0, 'f1_score': 0.0}
    
    def elastic_weight_consolidation(self, new_task_data):
        """弹性权重巩固"""
        # 保存当前最优参数
        self.optimal_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # 计算Fisher信息矩阵
        self.fisher_information = self.compute_fisher_information(new_task_data)
        
        self.logger.info("EWC setup completed")
    
    def compute_fisher_information(self, data_loader=None):
        """计算Fisher信息矩阵"""
        fisher = {}
        
        if data_loader is None:
            # 使用内存缓冲区的数据
            data_loader = self.memory_buffer.sample(batch_size=100)
        
        self.model.eval()
        
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        for batch in data_loader:
            self.model.zero_grad()
            
            if isinstance(batch, dict):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            else:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # 归一化
        for name in fisher:
            fisher[name] /= len(data_loader)
        
        return fisher
    
    def compute_ewc_loss(self):
        """计算EWC损失"""
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                ewc_loss += torch.sum(
                    self.fisher_information[name] * 
                    (param - self.optimal_params[name]) ** 2
                )
        return ewc_loss
    
    def save_checkpoint(self, filepath):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fisher_information': self.fisher_information,
            'optimal_params': self.optimal_params,
            'performance_history': dict(self.performance_monitor.metrics_history)
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fisher_information = checkpoint.get('fisher_information', {})
        self.optimal_params = checkpoint.get('optimal_params', {})
        
        # 恢复性能历史
        performance_history = checkpoint.get('performance_history', {})
        for key, values in performance_history.items():
            self.performance_monitor.metrics_history[key] = values
        
        self.logger.info(f"Checkpoint loaded from {filepath}")

# 使用示例
if __name__ == "__main__":
    # 假设有一个预训练模型
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained('bert-base-uncased')
    continual_system = ContinualLearningSystem(model)
    
    # 模拟数据流
    def simulate_data_stream():
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        texts = ["This is a sample text", "Another example", "More training data"]
        
        for text in texts:
            encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            yield encoded
    
    # 开始持续学习
    data_stream = simulate_data_stream()
    continual_system.online_learning(data_stream)