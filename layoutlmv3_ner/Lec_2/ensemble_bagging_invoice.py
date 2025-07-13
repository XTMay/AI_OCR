import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from sklearn.utils import resample
import json
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random

class BaggingInvoiceExtractor:
    """基于Bagging的发票信息提取器"""
    
    def __init__(self, base_model_name: str = 'microsoft/layoutlm-base-uncased', 
                 n_estimators: int = 5, random_state: int = 42):
        self.base_model_name = base_model_name
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.tokenizers = []
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
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
    
    def create_bootstrap_samples(self, dataset, sample_ratio: float = 1.0):
        """创建自助采样数据集"""
        bootstrap_datasets = []
        dataset_size = len(dataset)
        sample_size = int(dataset_size * sample_ratio)
        
        for i in range(self.n_estimators):
            # 有放回采样
            indices = np.random.choice(dataset_size, size=sample_size, replace=True)
            bootstrap_dataset = Subset(dataset, indices)
            bootstrap_datasets.append(bootstrap_dataset)
            
            self.logger.info(f"Bootstrap样本 {i+1}: {len(bootstrap_dataset)} 个样本")
        
        return bootstrap_datasets
    
    def train_base_models(self, bootstrap_datasets, val_dataset, epochs: int = 5):
        """训练基础模型"""
        for i, train_dataset in enumerate(bootstrap_datasets):
            self.logger.info(f"训练第 {i+1} 个基础模型...")
            
            # 初始化模型和分词器
            tokenizer = LayoutLMTokenizer.from_pretrained(self.base_model_name)
            model = LayoutLMForTokenClassification.from_pretrained(
                self.base_model_name,
                num_labels=len(self.label_to_id)
            )
            
            # 训练配置
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            
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
                self.logger.info(f"模型 {i+1}, Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
    
    def predict_ensemble(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """集成预测"""
        all_predictions = []
        
        # 每个模型进行预测
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            prediction = self._predict_single_model(model, tokenizer, text_blocks)
            all_predictions.append(prediction)
        
        # 投票聚合
        ensemble_result = self._aggregate_predictions(all_predictions)
        return ensemble_result
    
    def _predict_single_model(self, model, tokenizer, text_blocks):
        """单个模型预测"""
        model.eval()
        
        # 准备输入
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
        
        # 编码
        encoding = tokenizer(
            words,
            boxes=boxes,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 预测
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # 解析结果
        predicted_labels = [self.id_to_label[pred.item()] for pred in predictions[0]]
        entities = self._extract_entities(words, predicted_labels)
        
        return entities
    
    def _extract_entities(self, words, labels):
        """提取实体"""
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
    
    def _aggregate_predictions(self, all_predictions):
        """聚合多个预测结果"""
        # 统计每个字段的预测结果
        field_votes = defaultdict(list)
        
        for prediction in all_predictions:
            for field, value in prediction.items():
                if value:  # 只考虑非空预测
                    field_votes[field].append(value)
        
        # 多数投票
        final_result = {}
        for field, votes in field_votes.items():
            if votes:
                # 选择出现次数最多的预测
                vote_counts = defaultdict(int)
                for vote in votes:
                    vote_counts[vote] += 1
                
                final_result[field] = max(vote_counts.items(), key=lambda x: x[1])[0]
            else:
                final_result[field] = ""
        
        return final_result

# 使用示例
def demonstrate_bagging():
    """演示Bagging方法"""
    print("=== Bagging发票提取演示 ===")
    
    # 初始化Bagging提取器
    bagging_extractor = BaggingInvoiceExtractor(n_estimators=3)
    
    # 模拟训练数据（实际使用时替换为真实数据集）
    # train_dataset = YourInvoiceDataset(...)
    # val_dataset = YourValidationDataset(...)
    
    # 创建bootstrap样本
    # bootstrap_datasets = bagging_extractor.create_bootstrap_samples(train_dataset)
    
    # 训练基础模型
    # bagging_extractor.train_base_models(bootstrap_datasets, val_dataset)
    
    # 模拟预测
    sample_text_blocks = [
        {
            'text': 'Invoice No: INV-2024-001',
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 300, 'y_max': 70}
        },
        {
            'text': 'Date: 2024-01-15',
            'bbox': {'x_min': 100, 'y_min': 80, 'x_max': 250, 'y_max': 100}
        }
    ]
    
    # result = bagging_extractor.predict_ensemble(sample_text_blocks)
    # print(f"Bagging预测结果: {result}")
    
    print("Bagging方法的优势:")
    print("1. 减少过拟合，提高泛化能力")
    print("2. 对噪声数据具有鲁棒性")
    print("3. 可以并行训练，提高效率")