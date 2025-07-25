import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import List

class WarmupCosineAnnealingLR(_LRScheduler):
    """带预热的余弦退火学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 eta_min: float = 0, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * progress)) / 2 
                    for base_lr in self.base_lrs]

class AdaptiveLearningRateScheduler:
    """自适应学习率调度器"""
    
    def __init__(self, initial_lr: float = 3e-5):
        self.initial_lr = initial_lr
        self.patience = 3
        self.factor = 0.5
        self.min_lr = 1e-7
        self.best_metric = float('-inf')
        self.wait = 0
        self.current_lr = initial_lr
    
    def step(self, metric: float, optimizer) -> bool:
        """根据指标调整学习率"""
        improved = metric > self.best_metric
        
        if improved:
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                
                if self.current_lr < old_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.current_lr
                    
                    print(f"学习率从 {old_lr:.2e} 降低到 {self.current_lr:.2e}")
                    self.wait = 0
                    return True
        
        return False