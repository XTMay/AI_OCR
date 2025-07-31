import torch
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
from typing import Dict, List
import logging

class ProgressiveTrainingStrategy:
    """渐进式训练策略"""
    
    def __init__(self, model_name: str = 'microsoft/layoutlmv3-base'):
        self.model_name = model_name
        self.training_phases = [
            {
                "name": "预训练阶段",
                "epochs": 5,
                "learning_rate": 5e-5,
                "freeze_layers": ["embeddings", "encoder.layer.0", "encoder.layer.1"],
                "description": "冻结底层，只训练顶层分类器"
            },
            {
                "name": "微调阶段1",
                "epochs": 10,
                "learning_rate": 3e-5,
                "freeze_layers": ["embeddings"],
                "description": "解冻部分编码器层"
            },
            {
                "name": "微调阶段2",
                "epochs": 15,
                "learning_rate": 1e-5,
                "freeze_layers": [],
                "description": "全模型微调"
            }
        ]
    
    def setup_progressive_training(self, model: LayoutLMv3ForTokenClassification, phase: int) -> Dict:
        """设置渐进式训练参数"""
        if phase >= len(self.training_phases):
            raise ValueError(f"训练阶段 {phase} 超出范围")
        
        current_phase = self.training_phases[phase]
        
        # 冻结指定层
        self._freeze_layers(model, current_phase["freeze_layers"])
        
        # 设置训练参数
        training_config = {
            "learning_rate": current_phase["learning_rate"],
            "num_train_epochs": current_phase["epochs"],
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "dataloader_num_workers": 2,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_f1",
            "greater_is_better": True
        }
        
        logging.info(f"开始{current_phase['name']}: {current_phase['description']}")
        logging.info(f"学习率: {current_phase['learning_rate']}, 训练轮数: {current_phase['epochs']}")
        
        return training_config
    
    def _freeze_layers(self, model: nn.Module, freeze_layer_names: List[str]):
        """冻结指定层"""
        frozen_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            should_freeze = any(freeze_name in name for freeze_name in freeze_layer_names)
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += 1
            else:
                param.requires_grad = True
        
        logging.info(f"冻结参数: {frozen_params}/{total_params} ({frozen_params/total_params*100:.1f}%)")
    
    def get_phase_info(self, phase: int) -> Dict:
        """获取训练阶段信息"""
        if phase >= len(self.training_phases):
            return None
        return self.training_phases[phase]
    
    def get_total_phases(self) -> int:
        """获取总训练阶段数"""
        return len(self.training_phases)