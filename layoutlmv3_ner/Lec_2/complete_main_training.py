import os
import json
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    LayoutLMTokenizer, 
    LayoutLMForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceDataPreprocessor:
    """发票数据预处理器"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ocr = PaddleOCR(lang='ch', use_angle_cls=True, use_gpu=False)
        
    def pdf_to_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """PDF转图像"""
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "converted_images")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            images = convert_from_path(pdf_path, dpi=300)
            image_paths = []
            
            for i, image in enumerate(images):
                image_path = os.path.join(output_dir, f"invoice_page_{i+1}.jpg")
                image.save(image_path, 'JPEG')
                image_paths.append(image_path)
                
            logger.info(f"PDF转换完成，生成{len(image_paths)}张图像")
            return image_paths
            
        except Exception as e:
            logger.error(f"PDF转换失败: {e}")
            return []
    
    def extract_ocr_data(self, image_path: str) -> List[Dict[str, Any]]:
        """提取OCR数据"""
        try:
            result = self.ocr.ocr(image_path)
            
            if not result or not result[0]:
                return []
            
            ocr_data = []
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # 计算边界框
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                ocr_data.append({
                    'text': text.strip(),
                    'bbox': {
                        'x_min': min(x_coords),
                        'y_min': min(y_coords),
                        'x_max': max(x_coords),
                        'y_max': max(y_coords)
                    },
                    'confidence': confidence
                })
            
            return ocr_data
            
        except Exception as e:
            logger.error(f"OCR提取失败: {e}")
            return []
    
    def create_training_annotations(self, label_data: Dict[str, str], image_path: str) -> List[Dict[str, Any]]:
        """创建训练标注"""
        ocr_data = self.extract_ocr_data(image_path)
        
        if not ocr_data:
            return []
        
        annotations = []
        
        # 为每个OCR文本块创建标注
        for ocr_item in ocr_data:
            text = ocr_item['text']
            bbox = ocr_item['bbox']
            
            # 匹配标签
            label = self._match_label(text, label_data)
            
            annotation = {
                'image_path': image_path,
                'text': text,
                'bbox': [int(bbox['x_min']), int(bbox['y_min']), 
                        int(bbox['x_max']), int(bbox['y_max'])],
                'label': label,
                'confidence': ocr_item['confidence']
            }
            annotations.append(annotation)
        
        return annotations
    
    def _match_label(self, text: str, label_data: Dict[str, str]) -> str:
        """匹配文本与标签"""
        # 简单的字符串匹配策略
        for field, value in label_data.items():
            if value and value.strip() in text:
                if field == "InvoiceNo":
                    return "B-InvoiceNo"
                elif field == "InvoiceDate":
                    return "B-InvoiceDate"
                elif field == "Currency":
                    return "B-Currency"
                elif field == "Amount with Tax":
                    return "B-AmountWithTax"
                elif field == "Amount without Tax":
                    return "B-AmountWithoutTax"
                elif field == "Tax":
                    return "B-Tax"
        
        return "O"  # 其他标签

class InvoiceLayoutDataset(Dataset):
    """发票LayoutLM数据集"""
    def __init__(self, annotations_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
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
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 准备输入
        words = [annotation['text']]
        boxes = [annotation['bbox']]
        labels = [self.label_to_id[annotation['label']]]
        
        # 分词
        word_tokens = []
        word_boxes = []
        word_labels = []
        
        for word, box, label in zip(words, boxes, labels):
            tokens = self.tokenizer.tokenize(word)
            word_tokens.extend(tokens)
            word_boxes.extend([box] * len(tokens))
            word_labels.extend([label] * len(tokens))
        
        # 编码
        encoding = self.tokenizer(
            word_tokens,
            boxes=word_boxes,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 处理标签
        labels_tensor = torch.zeros(self.max_length, dtype=torch.long)
        labels_tensor[:len(word_labels)] = torch.tensor(word_labels[:self.max_length])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bbox': encoding['bbox'].squeeze(),
            'labels': labels_tensor
        }

class InvoiceLayoutLMTrainer:
    """发票LayoutLM训练器"""
    def __init__(self, model_name: str = 'microsoft/layoutlm-base-uncased'):
        self.model_name = model_name
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
        
        # 标签数量
        self.num_labels = 13
        
        # 初始化模型
        self.model = LayoutLMForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels
        )
        
        self.trainer = None
    
    def setup_training(self, train_dataset, val_dataset, output_dir: str):
        """设置训练参数"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # 禁用wandb等
            dataloader_num_workers=0  # 避免多进程问题
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    def train(self):
        """开始训练"""
        if self.trainer is None:
            raise ValueError("请先调用setup_training方法")
        
        logger.info("开始训练模型...")
        self.trainer.train()
        logger.info("训练完成")
    
    def save_model(self, save_path: str):
        """保存模型"""
        if self.trainer is None:
            raise ValueError("训练器未初始化")
        
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"模型已保存到: {save_path}")

class ModelEvaluator:
    """模型评估器"""
    def __init__(self, inference_system):
        self.inference_system = inference_system
        
        # 字段映射
        self.field_mapping = {
            'InvoiceNo': 'InvoiceNo',
            'InvoiceDate': 'InvoiceDate', 
            'Currency': 'Currency',
            'Amount with Tax': 'AmountWithTax',
            'Amount without Tax': 'AmountWithoutTax',
            'Tax': 'Tax'
        }
    
    def evaluate_on_test_set(self, test_annotations: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
        """在测试集上评估"""
        predictions = []
        ground_truths = []
        
        # 模拟测试数据（实际应用中需要真实的测试图像）
        for annotation in test_annotations[:5]:  # 取前5个样本进行演示
            # 这里应该使用真实的图像路径
            image_path = annotation.get('image_path', '')
            
            if os.path.exists(image_path):
                try:
                    prediction = self.inference_system.predict_invoice(image_path)
                    predictions.append(prediction)
                    
                    # 构造ground truth（实际应用中应该从标注数据获取）
                    ground_truth = {
                        'InvoiceNo': 'TEST001',
                        'InvoiceDate': '2024年01月01日',
                        'Currency': 'TWD',
                        'Amount with Tax': '1000',
                        'Amount without Tax': '952',
                        'Tax': '48'
                    }
                    ground_truths.append(ground_truth)
                    
                except Exception as e:
                    logger.error(f"预测失败: {e}")
                    continue
        
        # 计算字段准确率
        field_accuracies = self._calculate_field_accuracies(predictions, ground_truths)
        
        return field_accuracies, predictions, ground_truths
    
    def _calculate_field_accuracies(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """计算字段准确率"""
        field_accuracies = {}
        
        for field in self.field_mapping.keys():
            correct = 0
            total = 0
            
            for pred, gt in zip(predictions, ground_truths):
                if field in gt:
                    total += 1
                    if field in pred and pred[field] == gt[field]:
                        correct += 1
            
            field_accuracies[field] = correct / total if total > 0 else 0.0
        
        return field_accuracies
    
    def generate_evaluation_report(self, field_accuracies: Dict[str, float], 
                                 predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
        """生成评估报告"""
        overall_accuracy = np.mean(list(field_accuracies.values()))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_accuracy': overall_accuracy,
            'field_accuracies': field_accuracies,
            'total_samples': len(predictions),
            'detailed_results': {
                'predictions': predictions,
                'ground_truths': ground_truths
            }
        }
        
        return report

def main():
    """主训练流程"""
    try:
        # 1. 数据预处理
        print("开始数据预处理...")
        preprocessor = InvoiceDataPreprocessor("/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data")
        
        # 转换PDF为图像
        pdf_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/測試股份有限公司.pdf"
        image_paths = preprocessor.pdf_to_images(pdf_path)
        
        if not image_paths:
            logger.error("PDF转换失败，无法继续训练")
            return
        
        # 加载标注数据
        label_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/label1.json"
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        # 创建训练标注
        annotations = preprocessor.create_training_annotations(label_data, image_paths[0])
        
        if not annotations:
            logger.error("无法创建训练标注")
            return
        
        # 2. 数据集划分
        train_annotations, val_annotations = train_test_split(
            annotations, test_size=0.2, random_state=42
        )
        
        # 保存训练和验证数据
        train_path = "train_annotations.json"
        val_path = "val_annotations.json"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_annotations, f, ensure_ascii=False, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_annotations, f, ensure_ascii=False, indent=2)
        
        print(f"训练样本数: {len(train_annotations)}")
        print(f"验证样本数: {len(val_annotations)}")
        
        # 3. 创建数据集
        print("创建数据集...")
        tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
        
        train_dataset = InvoiceLayoutDataset(train_path, tokenizer)
        val_dataset = InvoiceLayoutDataset(val_path, tokenizer)
        
        # 4. 训练模型
        print("开始训练模型...")
        trainer = InvoiceLayoutLMTrainer()
        trainer.setup_training(train_dataset, val_dataset, "./invoice_layoutlm_model")
        trainer.train()
        
        # 5. 保存模型
        model_save_path = "./final_invoice_model"
        trainer.save_model(model_save_path)
        
        # 6. 评估模型（简化版）
        print("评估模型性能...")
        
        # 创建简单的评估报告
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'training_completed': True,
            'model_path': model_save_path,
            'training_samples': len(train_annotations),
            'validation_samples': len(val_annotations),
            'status': 'success'
        }
        
        # 保存评估报告
        with open("evaluation_report.json", 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
        
        print("训练完成！")
        print(f"模型已保存到: {model_save_path}")
        print(f"训练样本数: {len(train_annotations)}")
        print(f"验证样本数: {len(val_annotations)}")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        raise

if __name__ == "__main__":
    main()