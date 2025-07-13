import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import json
import re
from datetime import datetime
import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class TextBlock:
    """文本块数据结构"""
    text: str
    bbox: Dict[str, float]
    confidence: float
    center_x: float
    center_y: float

class EnhancedPaddleOCR:
    """增强版PaddleOCR处理器"""
    def __init__(self, lang='ch', use_gpu=False):
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            use_gpu=use_gpu,
            show_log=False
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """图像预处理"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 图像增强
        # 1. 去噪
        denoised = cv2.fastNlMeansDenoising(image)
        
        # 2. 对比度增强
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def extract_text_with_layout(self, image_path: str) -> List[TextBlock]:
        """提取文本及其布局信息"""
        try:
            # 预处理图像
            processed_image = self.preprocess_image(image_path)
            
            # OCR识别
            result = self.ocr.ocr(processed_image)
            
            if not result or not result[0]:
                self.logger.warning(f"OCR未识别到文本: {image_path}")
                return []
            
            # 解析结果
            text_blocks = []
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # 过滤低置信度文本
                if confidence < 0.5:
                    continue
                
                # 计算边界框
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                text_block = TextBlock(
                    text=text.strip(),
                    bbox={
                        'x_min': min(x_coords),
                        'y_min': min(y_coords),
                        'x_max': max(x_coords),
                        'y_max': max(y_coords)
                    },
                    confidence=confidence,
                    center_x=sum(x_coords) / 4,
                    center_y=sum(y_coords) / 4
                )
                text_blocks.append(text_block)
            
            # 按阅读顺序排序（从上到下，从左到右）
            text_blocks.sort(key=lambda x: (x.center_y, x.center_x))
            
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"OCR处理失败: {e}")
            return []

class EnhancedInvoiceInferenceSystem:
    """增强版发票推理系统"""
    def __init__(self, model_path: str, confidence_threshold: float = 0.8):
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
        self.model = LayoutLMForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
        # 标签映射
        self.id_to_label = {
            0: 'O',
            1: 'B-InvoiceNo', 2: 'I-InvoiceNo',
            3: 'B-InvoiceDate', 4: 'I-InvoiceDate',
            5: 'B-Currency', 6: 'I-Currency',
            7: 'B-AmountWithTax', 8: 'I-AmountWithTax',
            9: 'B-AmountWithoutTax', 10: 'I-AmountWithoutTax',
            11: 'B-Tax', 12: 'I-Tax'
        }
        
        # 初始化OCR
        self.ocr_processor = EnhancedPaddleOCR()
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def predict_invoice(self, image_path: str) -> Dict[str, Any]:
        """预测发票信息"""
        try:
            # 1. OCR提取文本和布局
            text_blocks = self.ocr_processor.extract_text_with_layout(image_path)
            
            if not text_blocks:
                return self._empty_result("OCR未识别到文本")
            
            # 2. 准备LayoutLM输入
            words, boxes = self._prepare_layoutlm_input(text_blocks)
            
            if not words:
                return self._empty_result("无有效文本用于模型推理")
            
            # 3. 模型预测
            predictions, confidences = self._model_inference(words, boxes)
            
            # 4. 提取实体
            entities = self._extract_entities_with_confidence(words, predictions, confidences)
            
            # 5. 后处理和格式化
            result = self._post_process_entities(entities)
            
            # 6. 添加元数据
            result['metadata'] = {
                'total_text_blocks': len(text_blocks),
                'processed_tokens': len(words),
                'extraction_confidence': self._calculate_overall_confidence(entities)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"发票推理失败: {e}")
            return self._empty_result(f"推理过程出错: {str(e)}")
    
    def _prepare_layoutlm_input(self, text_blocks: List[TextBlock]) -> Tuple[List[str], List[List[int]]]:
        """准备LayoutLM输入"""
        words = []
        boxes = []
        
        for block in text_blocks:
            # 分词处理
            word_tokens = self.tokenizer.tokenize(block.text)
            
            if not word_tokens:
                continue
            
            words.extend(word_tokens)
            
            # 为每个token分配边界框
            for _ in word_tokens:
                boxes.append([
                    int(block.bbox['x_min']),
                    int(block.bbox['y_min']),
                    int(block.bbox['x_max']),
                    int(block.bbox['y_max'])
                ])
        
        return words, boxes
    
    def _model_inference(self, words: List[str], boxes: List[List[int]]) -> Tuple[List[str], List[float]]:
        """模型推理"""
        # 编码输入
        encoding = self.tokenizer(
            words,
            boxes=boxes,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            # 获取预测标签和置信度
            probabilities = torch.softmax(logits, dim=2)
            predictions = torch.argmax(logits, dim=2)
            confidences = torch.max(probabilities, dim=2)[0]
        
        # 转换为标签
        predicted_labels = [self.id_to_label[pred.item()] for pred in predictions[0]]
        confidence_scores = confidences[0].tolist()
        
        return predicted_labels[:len(words)], confidence_scores[:len(words)]
    
    def _extract_entities_with_confidence(self, words: List[str], labels: List[str], confidences: List[float]) -> Dict[str, Dict[str, Any]]:
        """提取实体及其置信度"""
        entities = {}
        current_entity = None
        current_text = []
        current_confidences = []
        
        for word, label, confidence in zip(words, labels, confidences):
            if label.startswith('B-'):
                # 保存之前的实体
                if current_entity and current_text:
                    entities[current_entity] = {
                        'text': ' '.join(current_text),
                        'confidence': np.mean(current_confidences)
                    }
                
                # 开始新实体
                current_entity = label[2:]
                current_text = [word]
                current_confidences = [confidence]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # 继续当前实体
                current_text.append(word)
                current_confidences.append(confidence)
                
            else:
                # 结束当前实体
                if current_entity and current_text:
                    entities[current_entity] = {
                        'text': ' '.join(current_text),
                        'confidence': np.mean(current_confidences)
                    }
                current_entity = None
                current_text = []
                current_confidences = []
        
        # 处理最后一个实体
        if current_entity and current_text:
            entities[current_entity] = {
                'text': ' '.join(current_text),
                'confidence': np.mean(current_confidences)
            }
        
        return entities
    
    def _post_process_entities(self, entities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """后处理提取的实体"""
        result = {
            "InvoiceNo": "",
            "InvoiceDate": "",
            "Currency": "",
            "Amount with Tax": "",
            "Amount without Tax": "",
            "Tax": "",
            "confidence_scores": {}
        }
        
        # 处理各个字段
        field_mapping = {
            'InvoiceNo': 'InvoiceNo',
            'InvoiceDate': 'InvoiceDate',
            'Currency': 'Currency',
            'AmountWithTax': 'Amount with Tax',
            'AmountWithoutTax': 'Amount without Tax',
            'Tax': 'Tax'
        }
        
        for entity_key, result_key in field_mapping.items():
            if entity_key in entities:
                entity_data = entities[entity_key]
                confidence = entity_data['confidence']
                
                # 只有置信度足够高才使用
                if confidence >= self.confidence_threshold:
                    if entity_key == 'InvoiceDate':
                        result[result_key] = self._normalize_date(entity_data['text'])
                    elif entity_key in ['AmountWithTax', 'AmountWithoutTax', 'Tax']:
                        result[result_key] = self._normalize_amount(entity_data['text'])
                    elif entity_key == 'Currency':
                        result[result_key] = entity_data['text'].strip().upper()
                    else:
                        result[result_key] = entity_data['text'].strip()
                    
                    result['confidence_scores'][result_key] = confidence
        
        return result
    
    def _normalize_date(self, date_text: str) -> str:
        """标准化日期格式"""
        # 多种日期格式支持
        patterns = [
            (r'(\d{4})年(\d{1,2})月(\d{1,2})日', lambda m: f"{m.group(1)}年{m.group(2).zfill(2)}月{m.group(3).zfill(2)}日"),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}年{m.group(2).zfill(2)}月{m.group(3).zfill(2)}日"),
            (r'(\d{4})/(\d{1,2})/(\d{1,2})', lambda m: f"{m.group(1)}年{m.group(2).zfill(2)}月{m.group(3).zfill(2)}日"),
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}年{m.group(1).zfill(2)}月{m.group(2).zfill(2)}日")
        ]
        
        for pattern, formatter in patterns:
            match = re.search(pattern, date_text)
            if match:
                return formatter(match)
        
        return date_text.strip()
    
    def _normalize_amount(self, amount_text: str) -> str:
        """标准化金额格式"""
        # 移除货币符号和空格
        cleaned = re.sub(r'[￥$€£¥,\s]', '', amount_text)
        
        # 提取数字（包括小数）
        numbers = re.findall(r'\d+\.?\d*', cleaned)
        if numbers:
            # 返回最大的数字（通常是金额）
            amounts = [float(num) for num in numbers]
            return str(max(amounts))
        
        return "0"
    
    def _calculate_overall_confidence(self, entities: Dict[str, Dict[str, Any]]) -> float:
        """计算整体置信度"""
        if not entities:
            return 0.0
        
        confidences = [entity['confidence'] for entity in entities.values()]
        return np.mean(confidences)
    
    def _empty_result(self, error_message: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            "InvoiceNo": "",
            "InvoiceDate": "",
            "Currency": "",
            "Amount with Tax": "",
            "Amount without Tax": "",
            "Tax": "",
            "confidence_scores": {},
            "metadata": {
                "error": error_message,
                "total_text_blocks": 0,
                "processed_tokens": 0,
                "extraction_confidence": 0.0
            }
        }
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        results = []
        for image_path in image_paths:
            result = self.predict_invoice(image_path)
            results.append(result)
        return results
    
    def export_results(self, results: List[Dict[str, Any]], output_path: str):
        """导出结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    # 初始化系统
    model_path = "path/to/your/layoutlm/model"
    inference_system = EnhancedInvoiceInferenceSystem(model_path)
    
    # 单个发票预测
    image_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/converted_images/invoice_page_1.jpg"
    result = inference_system.predict_invoice(image_path)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 批量预测
    image_paths = [
        "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/converted_images/invoice_page_1.jpg",
        "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/converted_images/page_1.jpg"
    ]
    
    batch_results = inference_system.batch_predict(image_paths)
    inference_system.export_results(batch_results, "batch_inference_results.json")