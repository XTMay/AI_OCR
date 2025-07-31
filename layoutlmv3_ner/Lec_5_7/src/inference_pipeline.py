import os
import re
import json
import logging
import torch
from datetime import datetime
from typing import List, Dict, Any, Tuple
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
import easyocr  # 替换PaddleOCR导入

class InvoiceInferencePipeline:
    """增强的发票推理管道"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化推理管道
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 加载模型和处理器
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.processor = LayoutLMv3Processor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化OCR
        self.ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        
        # 标签映射
        self.id_to_label = {
            0: 'O',
            1: 'B-InvoiceNo',
            2: 'I-InvoiceNo', 
            3: 'B-InvoiceDate',
            4: 'I-InvoiceDate',
            5: 'B-Currency',
            6: 'I-Currency',
            7: 'B-AmountwithTax',
            8: 'I-AmountwithTax',
            9: 'B-AmountwithoutTax',
            10: 'I-AmountwithoutTax',
            11: 'B-Tax',
            12: 'I-Tax'
        }
        
        self.logger.info(f"Initialized inference pipeline on {self.device}")
    
    def process_invoice(self, pdf_path: str) -> Dict[str, str]:
        """
        处理发票PDF并提取信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的发票信息字典
        """
        try:
            # 1. PDF转图像
            image_paths = self._pdf_to_images(pdf_path)
            
            # 2. 处理每个页面
            all_predictions = []
            for image_path in image_paths:
                predictions = self._process_single_image(image_path)
                all_predictions.extend(predictions)
            
            # 3. 后处理和格式化
            result = self._post_process_predictions(all_predictions)
            
            self.logger.info(f"Successfully processed invoice: {pdf_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process invoice {pdf_path}: {str(e)}")
            raise
    
    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """将PDF转换为图像"""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            image_paths = []
            
            for i, image in enumerate(images):
                # 保存临时图像
                temp_path = f"/tmp/invoice_page_{i+1}.jpg"
                image.save(temp_path, 'JPEG', quality=95)
                image_paths.append(temp_path)
            
            return image_paths
            
        except Exception as e:
            self.logger.error(f"PDF conversion failed: {str(e)}")
            raise
    
    def _process_single_image(self, image_path: str) -> List[Dict[str, Any]]:
        """处理单个图像"""
        try:
            # 检查处理器配置
            processor_config = getattr(self.processor, 'image_processor', None)
            apply_ocr = getattr(processor_config, 'apply_ocr', False) if processor_config else False
            
            # 准备图像
            image = Image.open(image_path).convert('RGB')
            
            if apply_ocr:
                # 使用内置OCR的情况
                # 直接用图像进行推理，处理器会内部处理OCR
                predictions = self._run_inference(image, [], [])
                
                # 由于没有外部OCR结果，返回简化的结果
                results = []
                for i, pred_label in enumerate(predictions):
                    if pred_label != 'O':
                        results.append({
                            'text': f'token_{i}',  # 占位符文本
                            'bbox': [0, 0, 100, 20],  # 占位符边界框
                            'label': pred_label,
                            'confidence': 1.0
                        })
                return results
            else:
                # 使用外部OCR的情况
                # 1. OCR提取文本和位置
                ocr_results = self._extract_text_with_positions(image_path)
                
                if not ocr_results:
                    return []
                
                # 2. 准备LayoutLMv3输入
                words = [item['text'] for item in ocr_results]
                boxes = [self._normalize_box(item['bbox'], image.size) for item in ocr_results]
                
                # 3. 模型推理
                predictions = self._run_inference(image, words, boxes)
                
                # 4. 组合结果
                results = []
                for i, (word, box, pred_label) in enumerate(zip(words, boxes, predictions)):
                    if pred_label != 'O':  # 只保留有标签的预测
                        results.append({
                            'text': word,
                            'bbox': box,
                            'label': pred_label,
                            'confidence': ocr_results[i]['confidence']
                        })
                
                return results
            
        except Exception as e:
            self.logger.error(f"Image processing failed for {image_path}: {str(e)}")
            return []
    
    def _extract_text_with_positions(self, image_path: str) -> List[Dict[str, Any]]:
        """使用OCR提取文本和位置"""
        try:
            result = self.ocr.readtext(image_path)
            
            extracted_data = []
            for line in result if result else []:
                if len(line) >= 3:
                    bbox = line[0]
                    text = line[1]
                    confidence = line[2]
                    
                    # 标准化边界框
                    normalized_bbox = self._normalize_bbox(bbox)
                    
                    extracted_data.append({
                        'text': text,
                        'bbox': normalized_bbox,
                        'confidence': confidence
                    })
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return []
    
    def _normalize_bbox(self, bbox: List[List[float]]) -> List[int]:
        """标准化边界框格式"""
        if len(bbox) == 4 and all(len(point) == 2 for point in bbox):
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            return [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
        else:
            return [0, 0, 100, 20]
    
    def _normalize_box(self, bbox: List[int], image_size: Tuple[int, int]) -> List[int]:
        """将边界框标准化到1000x1000坐标系"""
        width, height = image_size
        x1, y1, x2, y2 = bbox
        
        # 标准化到1000x1000
        x1 = int((x1 / width) * 1000)
        y1 = int((y1 / height) * 1000)
        x2 = int((x2 / width) * 1000)
        y2 = int((y2 / height) * 1000)
        
        return [x1, y1, x2, y2]
    
    def _run_inference(self, image: Image.Image, words: List[str], boxes: List[List[int]]) -> List[str]:
        """运行LayoutLMv3推理"""
        try:
            # 检查处理器配置
            processor_config = getattr(self.processor, 'image_processor', None)
            apply_ocr = getattr(processor_config, 'apply_ocr', False) if processor_config else False
            
            # 根据配置决定如何调用处理器
            if apply_ocr:
                # 如果apply_ocr为True，不提供boxes参数
                encoding = self.processor(
                    image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            else:
                # 如果apply_ocr为False，提供words和boxes
                encoding = self.processor(
                    image,
                    words,
                    boxes=boxes,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            
            # 移动到设备
            for key in encoding:
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            # 解码预测结果
            predicted_labels = []
            if apply_ocr:
                # 当使用内置OCR时，返回所有预测结果
                for pred_id in predictions[0].cpu().numpy():
                    label = self.id_to_label.get(pred_id, 'O')
                    predicted_labels.append(label)
                # 过滤掉padding tokens和special tokens
                predicted_labels = [label for label in predicted_labels if label != 'O'][:len(words)]
            else:
                # 当使用外部OCR时，按words长度返回
                for i, pred_id in enumerate(predictions[0].cpu().numpy()):
                    if i < len(words):
                        label = self.id_to_label.get(pred_id, 'O')
                        predicted_labels.append(label)
            
            return predicted_labels[:len(words)] if not apply_ocr else predicted_labels
            
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            return ['O'] * len(words)
    
    def _post_process_predictions(self, predictions: List[Dict]) -> Dict[str, str]:
        """后处理预测结果，确保JSON格式一致"""
        entities = {
            "InvoiceNo": "",
            "InvoiceDate": "", 
            "Currency": "",
            "Amount with Tax": "",
            "Amount without Tax": "",
            "Tax": ""
        }
        
        # 标签映射
        label_mapping = {
            'InvoiceNo': 'InvoiceNo',
            'InvoiceDate': 'InvoiceDate',
            'Currency': 'Currency',
            'AmountwithTax': 'Amount with Tax',
            'AmountwithoutTax': 'Amount without Tax',
            'Tax': 'Tax'
        }
        
        # 合并同类实体
        for pred in predictions:
            label = pred['label'].replace('B-', '').replace('I-', '')
            if label in label_mapping:
                field_name = label_mapping[label]
                if entities[field_name]:
                    entities[field_name] += " " + pred['text']
                else:
                    entities[field_name] = pred['text']
        
        # 数据清洗和验证
        for field, value in entities.items():
            entities[field] = self._clean_and_validate_text(value, field)
        
        return entities
    
    def _clean_and_validate_text(self, text: str, field_type: str) -> str:
        """字段特定的数据清洗和验证"""
        if not text:
            return ""
        
        text = text.strip()
        
        if field_type == "InvoiceDate":
            return self._normalize_date(text)
        elif "Amount" in field_type or field_type == "Tax":
            return self._normalize_amount(text)
        elif field_type == "Currency":
            return self._normalize_currency(text)
        
        return text
    
    def _normalize_date(self, date_text: str) -> str:
        """标准化日期格式"""
        # 移除多余字符
        date_text = re.sub(r'[年月日]', '-', date_text)
        date_text = re.sub(r'[-/\s]+', '-', date_text)
        
        # 尝试解析并格式化
        try:
            # 支持多种日期格式
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y.%m.%d', '%d.%m.%Y']:
                try:
                    dt = datetime.strptime(date_text, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        except:
            pass
        
        return date_text
    
    def _normalize_amount(self, amount_text: str) -> str:
        """标准化金额格式"""
        # 移除货币符号和多余字符
        amount_text = re.sub(r'[¥$€£,，]', '', amount_text)
        amount_text = re.sub(r'[元圆]', '', amount_text)
        
        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', amount_text)
        if numbers:
            return numbers[0]
        
        return amount_text
    
    def _normalize_currency(self, currency_text: str) -> str:
        """标准化货币格式"""
        currency_mapping = {
            '人民币': 'RMB',
            '美元': 'USD',
            '欧元': 'EUR',
            '日元': 'JPY',
            '¥': 'RMB',
            '$': 'USD',
            '€': 'EUR'
        }
        
        currency_text = currency_text.strip().upper()
        
        for key, value in currency_mapping.items():
            if key in currency_text:
                return value
        
        return currency_text
    
    def batch_process(self, pdf_paths: List[str]) -> List[Dict[str, str]]:
        """批量处理多个发票"""
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_invoice(pdf_path)
                results.append({
                    'file': pdf_path,
                    'status': 'success',
                    'data': result
                })
            except Exception as e:
                results.append({
                    'file': pdf_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def evaluate_confidence(self, predictions: List[Dict]) -> Dict[str, float]:
        """评估预测置信度"""
        if not predictions:
            return {'overall': 0.0}
        
        confidences = [pred['confidence'] for pred in predictions]
        
        return {
            'overall': np.mean(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'std': np.std(confidences)
        }