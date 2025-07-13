from transformers import (
    LayoutLMForTokenClassification,
    LayoutLMTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

class InvoiceLayoutLMTrainer:
    def __init__(self, model_name='microsoft/layoutlm-base-uncased'):
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
        self.model = LayoutLMForTokenClassification.from_pretrained(
            model_name,
            num_labels=13  # 根据标签数量调整
        )
        
    def setup_training(self, train_dataset, val_dataset, output_dir):
        """设置训练参数"""
        training_args = TrainingArguments(
            output_dir=output_dir,                    # 模型输出目录
            num_train_epochs=10,                     # 训练轮数
            per_device_train_batch_size=4,           # 每个设备的训练批次大小
            per_device_eval_batch_size=4,            # 每个设备的验证批次大小
            warmup_steps=500,                        # 学习率预热步数
            weight_decay=0.01,                       # 权重衰减（L2正则化）
            logging_dir=f'{output_dir}/logs',         # 日志保存目录
            logging_steps=100,                       # 每多少步记录一次日志
            evaluation_strategy="epoch",             # 评估策略
            save_strategy="epoch",                   # 保存策略
            load_best_model_at_end=True,             # 训练结束时加载最佳模型
            metric_for_best_model="eval_f1",         # 最佳模型评判指标
            greater_is_better=True,                  # 指标越大越好
            save_total_limit=3,                      # 最多保存模型数量
            learning_rate=5e-5,                      # 学习率
            lr_scheduler_type="linear",              # 学习率调度器类型
            dataloader_num_workers=2                 # 数据加载器工作进程数
        )
        
        # 数据整理器
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
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
                if lab != -100:  # 忽略填充标签
                    true_predictions.append(pred.item())
                    true_labels.append(lab)
        
        # 计算F1分数
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        return {
            'f1': f1_score(true_labels, true_predictions, average='weighted'),
            'precision': precision_score(true_labels, true_predictions, average='weighted'),
            'recall': recall_score(true_labels, true_predictions, average='weighted')
        }
    
    def train(self):
        """开始训练"""
        self.trainer.train()
        
    def save_model(self, save_path):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)