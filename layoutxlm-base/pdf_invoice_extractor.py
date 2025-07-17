import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import json
import re
import os
from datetime import datetime
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
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

class PDFInvoiceExtractor:
    """PDF发票信息提取器 - OCR + LayoutXLM"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.8):
        # 初始化OCR - 使用最简化的参数配置
        self.ocr = PaddleOCR(
            lang='ch'
            # 移除所有可能不兼容的参数
        )
        
        # 初始化LayoutLM模型（如果有训练好的模型）
        if model_path and os.path.exists(model_path):
            self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
            self.model = LayoutLMForTokenClassification.from_pretrained(model_path)
            self.model.eval()
            self.use_layoutlm = True
        else:
            self.use_layoutlm = False
            print("警告: 未找到LayoutLM模型，将使用基于规则的提取方法")
        
        self.confidence_threshold = confidence_threshold
        
        # 标签映射（如果使用LayoutLM）
        self.id_to_label = {
            0: 'O',
            1: 'B-InvoiceNo',
            2: 'I-InvoiceNo',
            3: 'B-InvoiceDate',
            4: 'I-InvoiceDate',
            5: 'B-Currency',
            6: 'I-Currency',
            7: 'B-AmountWithTax',
            8: 'I-AmountWithTax',
            9: 'B-AmountWithoutTax',
            10: 'I-AmountWithoutTax',
            11: 'B-Tax',
            12: 'I-Tax'
        }
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def pdf_to_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """将PDF转换为图片"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), 'converted_images')
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 转换PDF为图片
            images = convert_from_path(pdf_path, dpi=300)
            image_paths = []
            
            for i, image in enumerate(images):
                image_path = os.path.join(output_dir, f'invoice_page_{i+1}.jpg')
                image.save(image_path, 'JPEG', quality=95)
                image_paths.append(image_path)
                self.logger.info(f"已保存页面 {i+1}: {image_path}")
            
            return image_paths
            
        except Exception as e:
            self.logger.error(f"PDF转换失败: {e}")
            return []
    
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
                
                # 计算边界框
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                bbox_dict = {
                    'x_min': min(x_coords),
                    'y_min': min(y_coords),
                    'x_max': max(x_coords),
                    'y_max': max(y_coords)
                }
                
                center_x = (bbox_dict['x_min'] + bbox_dict['x_max']) / 2
                center_y = (bbox_dict['y_min'] + bbox_dict['y_max']) / 2
                
                text_block = TextBlock(
                    text=text,
                    bbox=bbox_dict,
                    confidence=confidence,
                    center_x=center_x,
                    center_y=center_y
                )
                
                text_blocks.append(text_block)
            
            self.logger.info(f"OCR识别到 {len(text_blocks)} 个文本块")
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"OCR处理失败: {e}")
            return []
    
    def extract_invoice_with_layoutlm(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """使用LayoutLM提取发票信息"""
        if not self.use_layoutlm:
            return self.extract_invoice_with_rules(text_blocks)
        
        try:
            # 准备LayoutLM输入
            words, boxes = self._prepare_layoutlm_input(text_blocks)
            
            if not words:
                return self._empty_result("无有效文本用于模型推理")
            
            # 模型预测
            predictions, confidences = self._model_inference(words, boxes)
            
            # 提取实体
            entities = self._extract_entities_with_confidence(words, predictions, confidences)
            
            # 后处理和格式化
            result = self._post_process_entities(entities)
            
            return result
            
        except Exception as e:
            self.logger.error(f"LayoutLM推理失败: {e}")
            return self.extract_invoice_with_rules(text_blocks)
    
    def extract_invoice_with_rules(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """基于规则的发票信息提取（备用方案）"""
        # 将所有文本合并
        full_text = ' '.join([block.text for block in text_blocks])
        
        result = {
            "InvoiceNo": "",
            "InvoiceDate": "",
            "Currency": "USD",
            "Amount with Tax": "",
            "Amount without Tax": "",
            "Tax": ""
        }
        
        # 发票号码提取
        invoice_patterns = [
            r'[Yy]\s*(\d{12})',
            r'发票号码[：:]*\s*([A-Z0-9\s]+)',
            r'Invoice\s*No[.：:]*\s*([A-Z0-9\s]+)',
            r'(\d{12})'
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, full_text)
            if match:
                result["InvoiceNo"] = match.group(1).strip()
                break
        
        # 日期提取
        date_patterns = [
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{1,2})/(\d{1,2})/(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, full_text)
            if match:
                if '年' in pattern:
                    result["InvoiceDate"] = f"{match.group(1)}年{match.group(2).zfill(2)}月{match.group(3).zfill(2)}日"
                else:
                    result["InvoiceDate"] = match.group(0)
                break
        
        # 货币提取
        currency_patterns = [r'(USD|CNY|EUR|JPY)', r'美元|人民币|欧元|日元']
        for pattern in currency_patterns:
            match = re.search(pattern, full_text)
            if match:
                currency_map = {'美元': 'USD', '人民币': 'CNY', '欧元': 'EUR', '日元': 'JPY'}
                result["Currency"] = currency_map.get(match.group(1), match.group(1))
                break
        
        # 金额提取
        amount_patterns = [
            r'含税金额[：:]*\s*([\d,]+\.?\d*)',
            r'税前金额[：:]*\s*([\d,]+\.?\d*)',
            r'税额[：:]*\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)'
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    amounts.append(amount)
                except:
                    continue
        
        if amounts:
            amounts.sort(reverse=True)
            if len(amounts) >= 1:
                result["Amount with Tax"] = str(amounts[0])
            if len(amounts) >= 2:
                result["Amount without Tax"] = str(amounts[1])
            if len(amounts) >= 3:
                result["Tax"] = str(amounts[2])
            elif len(amounts) >= 2:
                result["Tax"] = str(amounts[0] - amounts[1])
        
        return result
    
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
        
        # 字段映射
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
    
    def _empty_result(self, error_message: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            "InvoiceNo": "",
            "InvoiceDate": "",
            "Currency": "",
            "Amount with Tax": "",
            "Amount without Tax": "",
            "Tax": "",
            "error": error_message
        }
    
    def extract_invoice_from_pdf(self, pdf_path: str, output_json_path: str = None) -> Dict[str, Any]:
        """从PDF提取发票信息的主函数"""
        self.logger.info(f"开始处理PDF文件: {pdf_path}")
        
        try:
            # 1. 转换PDF为图片
            image_paths = self.pdf_to_images(pdf_path)
            
            if not image_paths:
                return self._empty_result("PDF转换失败")
            
            # 2. 处理第一页（通常发票信息在第一页）
            image_path = image_paths[0]
            self.logger.info(f"处理图片: {image_path}")
            
            # 3. OCR提取文本和布局
            text_blocks = self.extract_text_with_layout(image_path)
            
            if not text_blocks:
                return self._empty_result("OCR未识别到文本")
            
            # 4. 使用LayoutLM或规则提取发票信息
            result = self.extract_invoice_with_layoutlm(text_blocks)
            
            # 5. 保存结果
            if output_json_path:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                self.logger.info(f"结果已保存到: {output_json_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            return self._empty_result(f"处理过程出错: {str(e)}")

# 使用示例
def main():
    # 初始化提取器
    # 如果有训练好的LayoutLM模型，请提供模型路径
    # model_path = "/path/to/your/layoutlm/model"
    extractor = PDFInvoiceExtractor(model_path=None)  # 使用基于规则的方法
    
    # PDF文件路径
    pdf_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/測試股份有限公司.pdf"
    
    # 输出JSON路径
    output_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/extracted_result.json"
    
    # 提取发票信息
    result = extractor.extract_invoice_from_pdf(pdf_path, output_path)
    
    # 打印结果
    print("\n=== 发票信息提取结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    return result

if __name__ == "__main__":
    main()