import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import json
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class AdaBoostInvoiceExtractor:
    """基于AdaBoost的发票信息提取器"""
    
    def __init__(self, base_model_name: str = 'microsoft/layoutlm-base-uncased',
                 n_estimators: int = 5, learning_rate: float = 1.0):
        self.base_model_name = base_model_name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []
        self.tokenizers = []
        
        # 标签映射
        self.label_to_id = {
            'O': 0,
            'B-InvoiceNo': 1, 'I-InvoiceNo': 2,
            'B-InvoiceDate': 3, 'I-InvoiceDate': 4,
            'B-Currency': 5, 'I-Currency': 6,
            'B-AmountWithTax': 7, 'I-AmountWithTax': 8,
            'B-AmountWithoutTax': 9, 'I-AmountWithoutTax': 10,
            'B-Tax': 11, 'I-Tax': 12
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_boosting_models(self, train_dataset, val_dataset, epochs_per_model: int = 3):
        """训练Boosting模型序列"""
        # 初始化样本权重
        sample_weights = np.ones(len(train_dataset)) / len(train_dataset)
        
        for i in range(self.n_estimators):
            self.logger.info(f"训练第 {i+1} 个Boosting模型...")
            
            # 根据样本权重创建加权采样器
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset, 
                batch_size=4, 
                sampler=sampler
            )
            
            # 初始化模型
            tokenizer = LayoutLMTokenizer.from_pretrained(self.base_model_name)
            model = LayoutLMForTokenClassification.from_pretrained(
                self.base_model_name,
                num_labels=len(self.label_to_id)
            )
            
            # 训练当前模型
            model = self._train_single_model(model, train_loader, epochs_per_model)
            
            # 计算模型在训练集上的错误率
            error_rate = self._calculate_error_rate(model, tokenizer, train_dataset, sample_weights)
            
            # 计算模型权重
            if error_rate == 0:
                model_weight = 1.0
            elif error_rate >= 0.5:
                model_weight = 0.1  # 给予很小的权重
            else:
                model_weight = 0.5 * np.log((1 - error_rate) / error_rate)
            
            # 更新样本权重
            sample_weights = self._update_sample_weights(
                model, tokenizer, train_dataset, sample_weights, model_weight
            )
            
            # 保存模型和权重
            self.models.append(model)
            self.model_weights.append(model_weight)
            self.tokenizers.append(tokenizer)
            
            self.logger.info(f"模型 {i+1}: 错误率 = {error_rate:.4f}, 权重 = {model_weight:.4f}")
    
    def _train_single_model(self, model, train_loader, epochs):
        """训练单个模型"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    bbox=batch['bbox'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return model
    
    def _calculate_error_rate(self, model, tokenizer, dataset, sample_weights):
        """计算加权错误率"""
        model.eval()
        total_weighted_error = 0
        total_weight = 0
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                # 预测
                outputs = model(
                    input_ids=sample['input_ids'].unsqueeze(0),
                    attention_mask=sample['attention_mask'].unsqueeze(0),
                    bbox=sample['bbox'].unsqueeze(0)
                )
                
                predictions = torch.argmax(outputs.logits, dim=2)
                true_labels = sample['labels']
                
                # 计算错误
                errors = (predictions.squeeze() != true_labels).float()
                weighted_error = torch.sum(errors) * sample_weights[i]
                
                total_weighted_error += weighted_error.item()
                total_weight += sample_weights[i]
        
        return total_weighted_error / total_weight if total_weight > 0 else 0
    
    def _update_sample_weights(self, model, tokenizer, dataset, sample_weights, model_weight):
        """更新样本权重"""
        model.eval()
        new_weights = sample_weights.copy()
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                # 预测
                outputs = model(
                    input_ids=sample['input_ids'].unsqueeze(0),
                    attention_mask=sample['attention_mask'].unsqueeze(0),
                    bbox=sample['bbox'].unsqueeze(0)
                )
                
                predictions = torch.argmax(outputs.logits, dim=2)
                true_labels = sample['labels']
                
                # 检查是否预测错误
                is_wrong = torch.any(predictions.squeeze() != true_labels).item()
                
                if is_wrong:
                    # 增加错误样本的权重
                    new_weights[i] *= np.exp(model_weight * self.learning_rate)
                else:
                    # 减少正确样本的权重
                    new_weights[i] *= np.exp(-model_weight * self.learning_rate)
        
        # 归一化权重
        new_weights = new_weights / np.sum(new_weights)
        return new_weights
    
    def predict_boosting(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Boosting集成预测"""
        weighted_predictions = defaultdict(lambda: defaultdict(float))
        
        # 每个模型进行加权预测
        for model, tokenizer, weight in zip(self.models, self.tokenizers, self.model_weights):
            prediction = self._predict_single_model(model, tokenizer, text_blocks)
            
            for field, value in prediction.items():
                if value:
                    weighted_predictions[field][value] += weight
        
        # 选择权重最高的预测
        final_result = {}
        for field, value_weights in weighted_predictions.items():
            if value_weights:
                final_result[field] = max(value_weights.items(), key=lambda x: x[1])[0]
            else:
                final_result[field] = ""
        
        return final_result
    
    def _predict_single_model(self, model, tokenizer, text_blocks):
        """单个模型预测（与Bagging中的实现相同）"""
        model.eval()
        
        words = []
        boxes = []
        
        for block in text_blocks:
            word_tokens = tokenizer.tokenize(block['text'])
            words.extend(word_tokens)
            
            for _ in word_tokens:
                boxes.append([
                    int(block['bbox']['x_min']),
                    int(block['bbox']['y_min']),
                    int(block['bbox']['x_max']),
                    int(block['bbox']['y_max'])
                ])
        
        encoding = tokenizer(
            words,
            boxes=boxes,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        predicted_labels = [self.id_to_label[pred.item()] for pred in predictions[0]]
        entities = self._extract_entities(words, predicted_labels)
        
        return entities
    
    def _extract_entities(self, words, labels):
        """提取实体（与Bagging中的实现相同）"""
        entities = {}
        current_entity = None
        current_text = []
        
        for word, label in zip(words, labels):
            if label.startswith('B-'):
                if current_entity and current_text:
                    entities[current_entity] = ' '.join(current_text)
                
                current_entity = label[2:]
                current_text = [word]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                current_text.append(word)
                
            else:
                if current_entity and current_text:
                    entities[current_entity] = ' '.join(current_text)
                current_entity = None
                current_text = []
        
        if current_entity and current_text:
            entities[current_entity] = ' '.join(current_text)
        
        return entities

# 使用示例
def demonstrate_boosting():
    """演示Boosting方法"""
    print("=== Boosting发票提取演示 ===")
    
    # 初始化Boosting提取器
    boosting_extractor = AdaBoostInvoiceExtractor(n_estimators=3)
    
    print("Boosting方法的优势:")
    print("1. 专注于困难样本，提高整体性能")
    print("2. 逐步减少偏差，提升准确率")
    print("3. 自适应调整模型权重")