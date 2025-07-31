import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging
from datetime import datetime

class TrainingMonitor:
    """训练监控系统"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_f1': [],
            'eval_precision': [],
            'eval_recall': [],
            'learning_rate': [],
            'epoch_time': []
        }
        self.start_time = None
        
    def start_training(self):
        """开始训练监控"""
        self.start_time = time.time()
        logging.info(f"开始训练监控: {datetime.now()}")
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float], lr: float):
        """记录每轮训练指标"""
        epoch_time = time.time() - self.start_time if self.start_time else 0
        
        # 更新历史记录
        self.metrics_history['train_loss'].append(metrics.get('train_loss', 0))
        self.metrics_history['eval_loss'].append(metrics.get('eval_loss', 0))
        self.metrics_history['eval_f1'].append(metrics.get('eval_f1', 0))
        self.metrics_history['eval_precision'].append(metrics.get('eval_precision', 0))
        self.metrics_history['eval_recall'].append(metrics.get('eval_recall', 0))
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['epoch_time'].append(epoch_time)
        
        # 打印当前指标
        logging.info(f"Epoch {epoch}:")
        logging.info(f"  Train Loss: {metrics.get('train_loss', 0):.4f}")
        logging.info(f"  Eval Loss: {metrics.get('eval_loss', 0):.4f}")
        logging.info(f"  F1 Score: {metrics.get('eval_f1', 0):.4f}")
        logging.info(f"  Precision: {metrics.get('eval_precision', 0):.4f}")
        logging.info(f"  Recall: {metrics.get('eval_recall', 0):.4f}")
        logging.info(f"  Learning Rate: {lr:.2e}")
        
        # 保存指标到文件
        self._save_metrics()
    
    def _save_metrics(self):
        """保存指标到文件"""
        metrics_file = f"{self.log_dir}/training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.metrics_history['eval_loss'], label='Eval Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # F1分数曲线
        axes[0, 1].plot(self.metrics_history['eval_f1'], label='F1 Score', color='green')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 精确率和召回率
        axes[1, 0].plot(self.metrics_history['eval_precision'], label='Precision', color='orange')
        axes[1, 0].plot(self.metrics_history['eval_recall'], label='Recall', color='purple')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        axes[1, 1].plot(self.metrics_history['learning_rate'], label='Learning Rate', color='brown')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_training_report(self) -> Dict[str, Any]:
        """生成训练报告"""
        if not self.metrics_history['eval_f1']:
            return {"error": "没有训练数据"}
        
        best_epoch = self.metrics_history['eval_f1'].index(max(self.metrics_history['eval_f1']))
        
        report = {
            "training_summary": {
                "total_epochs": len(self.metrics_history['eval_f1']),
                "best_epoch": best_epoch + 1,
                "best_f1_score": max(self.metrics_history['eval_f1']),
                "final_f1_score": self.metrics_history['eval_f1'][-1],
                "total_training_time": sum(self.metrics_history['epoch_time'])
            },
            "best_metrics": {
                "f1_score": self.metrics_history['eval_f1'][best_epoch],
                "precision": self.metrics_history['eval_precision'][best_epoch],
                "recall": self.metrics_history['eval_recall'][best_epoch],
                "eval_loss": self.metrics_history['eval_loss'][best_epoch]
            },
            "convergence_analysis": {
                "converged": self._check_convergence(),
                "overfitting_detected": self._detect_overfitting(),
                "learning_rate_adjustments": self._count_lr_adjustments()
            }
        }
        
        return report
    
    def _check_convergence(self, patience: int = 5, threshold: float = 0.001) -> bool:
        """检查是否收敛"""
        if len(self.metrics_history['eval_f1']) < patience:
            return False
        
        recent_scores = self.metrics_history['eval_f1'][-patience:]
        return max(recent_scores) - min(recent_scores) < threshold
    
    def _detect_overfitting(self, lookback: int = 5) -> bool:
        """检测过拟合"""
        if len(self.metrics_history['train_loss']) < lookback * 2:
            return False
        
        recent_train_loss = self.metrics_history['train_loss'][-lookback:]
        recent_eval_loss = self.metrics_history['eval_loss'][-lookback:]
        
        # 训练损失持续下降但验证损失上升
        train_trend = recent_train_loss[-1] < recent_train_loss[0]
        eval_trend = recent_eval_loss[-1] > recent_eval_loss[0]
        
        return train_trend and eval_trend
    
    def _count_lr_adjustments(self) -> int:
        """统计学习率调整次数"""
        adjustments = 0
        for i in range(1, len(self.metrics_history['learning_rate'])):
            if self.metrics_history['learning_rate'][i] != self.metrics_history['learning_rate'][i-1]:
                adjustments += 1
        return adjustments