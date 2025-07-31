import torch
from torch.utils.data import Dataset
from transformers import (
    LayoutLMv3Tokenizer,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import json
from PIL import Image
from typing import Dict, List
import logging

class InvoiceLayoutDataset(Dataset):
    """发票LayoutLMv3数据集"""
    
    def __init__(self, annotations_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 更新标签映射，与label1.json字段对应
        self.label_to_id = {
            'O': 0,
            'B-InvoiceNo': 1, 'I-InvoiceNo': 2,
            'B-InvoiceDate': 3, 'I-InvoiceDate': 4,
            'B-Currency': 5, 'I-Currency': 6,
            'B-AmountwithTax': 7, 'I-AmountwithTax': 8,  # 注意：移除空格
            'B-AmountwithoutTax': 9, 'I-AmountwithoutTax': 10,
            'B-Tax': 11, 'I-Tax': 12
        }
        
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 加载图像
        image = Image.open(annotation['image_path']).convert('RGB')
        
        # 准备文本和边界框
        words = []
        boxes = []
        labels = []
        
        for item in annotation['entities']:
            # 分词
            word_tokens = self.tokenizer.tokenize(item['text'])
            words.extend(word_tokens)
            
            # 为每个token分配边界框和标签
            for i, token in enumerate(word_tokens):
                boxes.append(item['bbox'])
                # BIO标注
                if i == 0:
                    labels.append(self.label_to_id[item['label']])
                else:
                    # I-标签
                    i_label = item['label'].replace('B-', 'I-')
                    labels.append(self.label_to_id.get(i_label, self.label_to_id['O']))
        
        # 编码
        encoding = self.tokenizer(
            words,
            boxes=boxes,
            word_labels=labels,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bbox': encoding['bbox'].squeeze(0),  # Remove batch dimension but keep [seq_len, 4] shape
            'labels': encoding['labels'].flatten(),
            'pixel_values': encoding.get('pixel_values', torch.zeros(3, 224, 224))
        }

class LayoutLMv3Trainer:
    """LayoutLMv3训练器"""
    
    def __init__(self, model_name: str = 'microsoft/layoutlmv3-base'):
        self.model_name = model_name
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            num_labels=13  # 根据标签数量调整
        )
        
        # Also load the processor for complete model saving
        try:
            from transformers import LayoutLMv3Processor
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        except Exception as e:
            logging.warning(f"Could not load processor: {e}, will use tokenizer only")
        
    def setup_training(self, train_dataset, val_dataset, output_dir: str, num_epochs: int = 15):
        """设置训练参数"""
        # 检测设备类型，决定是否使用fp16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_fp16 = torch.cuda.is_available()  # 只在CUDA设备上使用fp16
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,  # 适合LayoutLMv3的批次大小
            per_device_eval_batch_size=2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="epoch",  # 修改：从 evaluation_strategy 改为 eval_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,
            learning_rate=3e-5,  # LayoutLMv3推荐学习率
            lr_scheduler_type="cosine",
            dataloader_num_workers=2,
            fp16=use_fp16,  # 只在CUDA设备上使用混合精度训练
            gradient_accumulation_steps=4,  # 梯度累积
            report_to=None  # 禁用wandb等
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = torch.argmax(torch.tensor(predictions), dim=2)
        
        # 移除填充的标签
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred, lab in zip(prediction, label):
                if lab != -100:
                    true_predictions.append(pred.item())
                    true_labels.append(lab)
        
        # 计算指标
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        return {
            'f1': f1_score(true_labels, true_predictions, average='weighted'),
            'precision': precision_score(true_labels, true_predictions, average='weighted'),
            'recall': recall_score(true_labels, true_predictions, average='weighted')
        }
    
    def train(self):
        """开始训练"""
        self.trainer.train()
    
    def save_model(self, save_path: str):
        """保存模型及所有相关组件"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save processor if available
        if hasattr(self, 'processor'):
            try:
                self.processor.save_pretrained(save_path)
                logging.info(f"✅ 模型、tokenizer和processor已保存到: {save_path}")
            except Exception as e:
                logging.warning(f"⚠️ 无法保存processor: {e}")
                logging.info(f"✅ 模型和tokenizer已保存到: {save_path}")
        else:
            # If no processor, create a minimal preprocessor_config.json
            import json
            preprocessor_config = {
                "do_resize": True,
                "size": {"height": 224, "width": 224},
                "do_normalize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "apply_ocr": False,
                "ocr_lang": None,
                "processor_class": "LayoutLMv3Processor"
            }
            
            config_path = os.path.join(save_path, "preprocessor_config.json")
            with open(config_path, 'w') as f:
                json.dump(preprocessor_config, f, indent=2)
            
            logging.info(f"✅ 模型、tokenizer和preprocessor_config已保存到: {save_path}")