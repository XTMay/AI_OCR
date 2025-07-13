import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import pickle

class StackingInvoiceExtractor:
    """基于Stacking的发票信息提取器"""
    
    def __init__(self, base_model_names: List[str] = None, 
                 meta_learner_type: str = 'rf', cv_folds: int = 5):
        if base_model_names is None:
            self.base_model_names = [
                'microsoft/layoutlm-base-uncased',
                'microsoft/layoutlm-base-uncased',  # 可以用不同的超参数
                'microsoft/layoutlm-base-uncased'   # 或不同的预处理方式
            ]
        else:
            self.base_model_names = base_model_names
        
        self.meta_learner_type = meta_learner_type
        self.cv_folds = cv_folds
        self.base_models = []
        self.tokenizers = []
        self.meta_learners = {}  # 每个字段一个元学习器
        
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
        
        # 发票字段
        self.invoice_fields = ['InvoiceNo', 'InvoiceDate', 'Currency', 
                              'AmountWithTax', 'AmountWithoutTax', 'Tax']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_stacking_models(self, train_dataset, val_dataset, epochs_per_model: int = 5):
        """训练Stacking模型"""
        # 第一层：训练基础模型
        self.logger.info("训练基础模型...")
        self._train_base_models(train_dataset, val_dataset, epochs_per_model)
        
        # 第二层：生成元特征并训练元学习器
        self.logger.info("生成元特征并训练元学习器...")
        self._train_meta_learners(train_dataset)
    
    def _train_base_models(self, train_dataset, val_dataset, epochs):
        """训练基础模型"""
        for i, model_name in enumerate(self.base_model_names):
            self.logger.info(f"训练基础模型 {i+1}/{len(self.base_model_names)}...")
            
            # 初始化模型和分词器
            tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
            model = LayoutLMForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(self.label_to_id)
            )
            
            # 可以为不同的基础模型设置不同的超参数
            if i == 0:
                lr = 2e-5
            elif i == 1:
                lr = 1e-5
            else:
                lr = 3e-5
            
            # 训练模型
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
                self.logger.info(f"基础模型 {i+1}, Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            self.base_models.append(model)
            self.tokenizers.append(tokenizer)
    
    def _train_meta_learners(self, train_dataset):
        """训练元学习器"""
        # 使用交叉验证生成元特征
        meta_features, meta_labels = self._generate_meta_features(train_dataset)
        
        # 为每个字段训练元学习器
        for field in self.invoice_fields:
            if field in meta_labels and len(meta_labels[field]) > 0:
                self.logger.info(f"训练字段 {field} 的元学习器...")
                
                X = meta_features
                y = meta_labels[field]
                
                # 初始化元学习器
                if self.meta_learner_type == 'rf':
                    meta_learner = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=42,
                        max_depth=10
                    )
                elif self.meta_learner_type == 'lr':
                    meta_learner = LogisticRegression(
                        random_state=42,
                        max_iter=1000
                    )
                else:
                    raise ValueError(f"不支持的元学习器类型: {self.meta_learner_type}")
                
                # 训练元学习器
                meta_learner.fit(X, y)
                self.meta_learners[field] = meta_learner
                
                self.logger.info(f"字段 {field} 元学习器训练完成")
    
    def _generate_meta_features(self, dataset):
        """生成元特征"""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        meta_features = []
        meta_labels = defaultdict(list)
        
        # 交叉验证生成元特征
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
            self.logger.info(f"生成第 {fold+1} 折元特征...")
            
            # 获取验证集样本的基础模型预测
            for val_i in val_idx:
                sample = dataset[val_i]
                
                # 模拟文本块（实际应用中需要从样本中提取）
                text_blocks = self._extract_text_blocks_from_sample(sample)
                
                # 获取所有基础模型的预测
                base_predictions = []
                for model, tokenizer in zip(self.base_models, self.tokenizers):
                    pred = self._predict_single_model(model, tokenizer, text_blocks)
                    base_predictions.append(pred)
                
                # 构造元特征向量
                feature_vector = self._construct_feature_vector(base_predictions)
                meta_features.append(feature_vector)
                
                # 构造元标签（实际应用中需要从真实标注获取）
                true_labels = self._extract_true_labels_from_sample(sample)
                for field, label in true_labels.items():
                    meta_labels[field].append(label)
        
        return np.array(meta_features), meta_labels
    
    def _extract_text_blocks_from_sample(self, sample):
        """从样本中提取文本块（模拟实现）"""
        # 这里需要根据实际的数据格式来实现
        # 模拟返回
        return [
            {
                'text': 'Sample text',
                'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 300, 'y_max': 70}
            }
        ]
    
    def _extract_true_labels_from_sample(self, sample):
        """从样本中提取真实标签（模拟实现）"""
        # 这里需要根据实际的标注格式来实现
        # 模拟返回
        return {
            'InvoiceNo': 'INV-001',
            'InvoiceDate': '2024-01-01',
            'Currency': 'USD'
        }
    
    def _construct_feature_vector(self, base_predictions):
        """构造元特征向量"""
        features = []
        
        # 为每个字段和每个基础模型创建特征
        for field in self.invoice_fields:
            for pred in base_predictions:
                # 字段是否被预测到（二进制特征）
                features.append(1 if field in pred and pred[field] else 0)
                
                # 预测值的长度（数值特征）
                if field in pred and pred[field]:
                    features.append(len(pred[field]))
                else:
                    features.append(0)
        
        # 基础模型之间的一致性特征
        for field in self.invoice_fields:
            values = [pred.get(field, '') for pred in base_predictions]
            # 计算一致性（相同预测的比例）
            if values:
                most_common = max(set(values), key=values.count)
                consistency = values.count(most_common) / len(values)
                features.append(consistency)
            else:
                features.append(0)
        
        return features
    
    def _predict_single_model(self, model, tokenizer, text_blocks):
        """单个模型预测（与前面实现相同）"""
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
        """提取实体（与前面实现相同）"""
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
    
    def predict_stacking(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stacking集成预测"""
        # 获取所有基础模型的预测
        base_predictions = []
        for model, tokenizer in zip(self.base_models, self.tokenizers):
            pred = self._predict_single_model(model, tokenizer, text_blocks)
            base_predictions.append(pred)
        
        # 构造元特征
        feature_vector = self._construct_feature_vector(base_predictions)
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # 使用元学习器进行最终预测
        final_result = {}
        for field in self.invoice_fields:
            if field in self.meta_learners:
                # 元学习器预测
                meta_pred = self.meta_learners[field].predict(feature_vector)[0]
                final_result[field] = meta_pred
            else:
                # 如果没有元学习器，使用多数投票
                votes = [pred.get(field, '') for pred in base_predictions]
                if votes:
                    final_result[field] = max(set(votes), key=votes.count)
                else:
                    final_result[field] = ''
        
        return final_result
    
    def save_models(self, save_dir: str):
        """保存所有模型"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存基础模型
        for i, (model, tokenizer) in enumerate(zip(self.base_models, self.tokenizers)):
            model_dir = os.path.join(save_dir, f'base_model_{i}')
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        
        # 保存元学习器
        meta_learners_path = os.path.join(save_dir, 'meta_learners.pkl')
        with open(meta_learners_path, 'wb') as f:
            pickle.dump(self.meta_learners, f)
        
        self.logger.info(f"所有模型已保存到: {save_dir}")

# 使用示例
def demonstrate_stacking():
    """演示Stacking方法"""
    print("=== Stacking发票提取演示 ===")
    
    # 初始化Stacking提取器
    stacking_extractor = StackingInvoiceExtractor(
        meta_learner_type='rf',
        cv_folds=3
    )
    
    print("Stacking方法的优势:")
    print("1. 学习最优的模型组合方式")
    print("2. 充分利用基础模型的互补性")
    print("3. 通常能获得最佳的集成性能")