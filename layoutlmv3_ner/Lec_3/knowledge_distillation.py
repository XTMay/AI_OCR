import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """知识蒸馏损失函数"""
        # 软标签损失
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 组合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss
    
    def train_step(self, batch):
        """训练步骤"""
        # 教师模型推理
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch)
            teacher_logits = teacher_outputs['logits']
        
        # 学生模型推理
        student_outputs = self.student_model(**batch)
        student_logits = student_outputs['logits']
        
        # 计算蒸馏损失
        loss = self.distillation_loss(
            student_logits, 
            teacher_logits, 
            batch['labels']
        )
        
        return loss

class ModelPruning:
    def __init__(self, model, pruning_ratio=0.2):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def magnitude_based_pruning(self):
        """基于权重大小的剪枝"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 计算权重的重要性
                importance = torch.abs(module.weight.data)
                
                # 确定剪枝阈值
                threshold = torch.quantile(
                    importance.flatten(), 
                    self.pruning_ratio
                )
                
                # 应用剪枝掩码
                mask = importance > threshold
                module.weight.data *= mask.float()
    
    def structured_pruning(self):
        """结构化剪枝"""
        # 移除整个神经元或通道
        pass