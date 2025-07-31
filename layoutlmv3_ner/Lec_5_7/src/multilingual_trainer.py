import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LayoutLMv3Tokenizer, 
    LayoutLMv3ForTokenClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
from collections import defaultdict

class MultilingualInvoiceTrainer:
    """多语言发票训练器"""
    
    def __init__(self):
        # 使用多语言预训练模型
        self.model_name = "microsoft/layoutlmv3-base"
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(self.model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 多语言标签映射
        self.label_mapping = {
            'O': 0,
            'B-INVOICE_NO': 1, 'I-INVOICE_NO': 2,
            'B-INVOICE_DATE': 3, 'I-INVOICE_DATE': 4,
            'B-CURRENCY': 5, 'I-CURRENCY': 6,
            'B-AMOUNT_WITH_TAX': 7, 'I-AMOUNT_WITH_TAX': 8,
            'B-AMOUNT_WITHOUT_TAX': 9, 'I-AMOUNT_WITHOUT_TAX': 10,
            'B-TAX': 11, 'I-TAX': 12,
            'B-COMPANY_NAME': 13, 'I-COMPANY_NAME': 14,
            'B-ADDRESS': 15, 'I-ADDRESS': 16
        }
        
        self.id2label = {v: k for k, v in self.label_mapping.items()}
        self.num_labels = len(self.label_mapping)
        
        # 支持的语言
        self.supported_languages = ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'es']
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_multilingual_training(self):
        """设置多语言训练参数"""
        training_args = TrainingArguments(
            output_dir="./models/multilingual_invoice",
            num_train_epochs=10,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            # 学习率调度
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            # 梯度累积
            gradient_accumulation_steps=2,
            # 混合精度训练
            fp16=True,
            # 早停
            save_total_limit=3,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard"
        )
        return training_args
    
    def initialize_model(self):
        """初始化多语言模型"""
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label_mapping
        )
        self.model.to(self.device)
        self.logger.info(f"Model initialized on {self.device}")
        
    def prepare_multilingual_data(self, data_paths: Dict[str, str]) -> Tuple[Dataset, Dataset]:
        """准备多语言训练数据"""
        all_train_data = []
        all_val_data = []
        
        for lang, data_path in data_paths.items():
            if lang not in self.supported_languages:
                self.logger.warning(f"Language {lang} not supported, skipping...")
                continue
                
            train_data, val_data = self._load_language_data(data_path, lang)
            all_train_data.extend(train_data)
            all_val_data.extend(val_data)
            
            self.logger.info(f"Loaded {len(train_data)} train and {len(val_data)} val samples for {lang}")
        
        # 数据平衡
        all_train_data = self._balance_multilingual_data(all_train_data)
        
        train_dataset = MultilingualInvoiceDataset(all_train_data, self.tokenizer, self.label_mapping)
        val_dataset = MultilingualInvoiceDataset(all_val_data, self.tokenizer, self.label_mapping)
        
        return train_dataset, val_dataset
    
    def _load_language_data(self, data_path: str, language: str) -> Tuple[List, List]:
        """加载特定语言的数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 为每个样本添加语言标识
        for item in data:
            item['language'] = language
        
        # 分割训练和验证数据
        split_idx = int(len(data) * 0.8)
        return data[:split_idx], data[split_idx:]
    
    def _balance_multilingual_data(self, data: List) -> List:
        """平衡多语言数据"""
        # 按语言分组
        lang_groups = defaultdict(list)
        for item in data:
            lang_groups[item['language']].append(item)
        
        # 找到最大样本数
        max_samples = max(len(samples) for samples in lang_groups.values())
        
        # 对少数语言进行上采样
        balanced_data = []
        for lang, samples in lang_groups.items():
            if len(samples) < max_samples:
                # 重复采样到最大数量
                multiplier = max_samples // len(samples)
                remainder = max_samples % len(samples)
                
                balanced_samples = samples * multiplier + samples[:remainder]
            else:
                balanced_samples = samples
            
            balanced_data.extend(balanced_samples)
        
        return balanced_data
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # 移除忽略的标签
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # 展平列表
        flat_true_predictions = [item for sublist in true_predictions for item in sublist]
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_true_labels, flat_true_predictions, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(flat_true_labels, flat_true_predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_multilingual_model(self, data_paths: Dict[str, str]):
        """训练多语言模型"""
        self.logger.info("Starting multilingual training...")
        
        # 初始化模型
        self.initialize_model()
        
        # 准备数据
        train_dataset, val_dataset = self.prepare_multilingual_data(data_paths)
        
        # 设置训练参数
        training_args = self.setup_multilingual_training()
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最佳模型
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        self.logger.info("Training completed!")
        
        return trainer
    
    def evaluate_on_language(self, test_data_path: str, language: str) -> Dict:
        """在特定语言上评估模型"""
        if not self.model:
            raise ValueError("Model not initialized. Please train or load a model first.")
        
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 添加语言标识
        for item in test_data:
            item['language'] = language
        
        test_dataset = MultilingualInvoiceDataset(test_data, self.tokenizer, self.label_mapping)
        
        # 创建评估器
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # 评估
        results = trainer.evaluate(test_dataset)
        
        self.logger.info(f"Evaluation results for {language}: {results}")
        return results
    
    def cross_lingual_transfer(self, source_lang: str, target_lang: str, 
                              source_data_path: str, target_data_path: str):
        """跨语言迁移学习"""
        self.logger.info(f"Starting cross-lingual transfer from {source_lang} to {target_lang}")
        
        # 首先在源语言上训练
        source_data = {source_lang: source_data_path}
        self.train_multilingual_model(source_data)
        
        # 在目标语言上微调
        target_data = {target_lang: target_data_path}
        
        # 使用较小的学习率进行微调
        fine_tune_args = self.setup_multilingual_training()
        fine_tune_args.learning_rate = 1e-5
        fine_tune_args.num_train_epochs = 3
        fine_tune_args.output_dir = f"./models/transfer_{source_lang}_to_{target_lang}"
        
        train_dataset, val_dataset = self.prepare_multilingual_data(target_data)
        
        trainer = Trainer(
            model=self.model,
            args=fine_tune_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        trainer.save_model()
        
        self.logger.info(f"Cross-lingual transfer completed: {source_lang} -> {target_lang}")
        
    def save_model(self, save_path: str):
        """保存模型"""
        if not self.model:
            raise ValueError("No model to save")
        
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存配置
        config = {
            'model_name': self.model_name,
            'label_mapping': self.label_mapping,
            'supported_languages': self.supported_languages
        }
        
        with open(os.path.join(save_path, 'trainer_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        # 加载配置
        config_path = os.path.join(model_path, 'trainer_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.label_mapping = config.get('label_mapping', self.label_mapping)
            self.supported_languages = config.get('supported_languages', self.supported_languages)
        
        self.logger.info(f"Model loaded from {model_path}")


class MultilingualInvoiceDataset(Dataset):
    """多语言发票数据集"""
    
    def __init__(self, data: List, tokenizer, label_mapping: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 提取文本和标签
        words = item['words']
        labels = item['labels']
        boxes = item['boxes']
        
        # 标记化
        encoding = self.tokenizer(
            words,
            boxes=boxes,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt',
            is_split_into_words=True
        )
        
        # 对齐标签
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label_mapping.get(labels[word_idx], 0))
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bbox': encoding['bbox'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }