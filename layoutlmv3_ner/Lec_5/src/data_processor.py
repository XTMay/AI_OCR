import os
import re
import json
import logging
from typing import List, Dict, Tuple, Any
from pdf2image import convert_from_path
import easyocr  # 替换PaddleOCR
from PIL import Image
import numpy as np
from pathlib import Path

class InvoiceDataProcessor:
    """优化的中英文发票数据预处理器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # 使用EasyOCR替代PaddleOCR，支持中英文
        self.ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        
        # 标准化标签映射
        self.target_fields = {
            "InvoiceNo": "发票号码",
            "InvoiceDate": "发票日期", 
            "Currency": "货币",
            "Amount with Tax": "含税金额",
            "Amount without Tax": "不含税金额",
            "Tax": "税额"
        }
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        dirs = ['processed', 'training', 'validation', 'images']
        for dir_name in dirs:
            dir_path = os.path.join(self.data_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
    
    def pdf_to_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """将PDF转换为高质量图像"""
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'images')
        
        try:
            # 高质量转换设置
            images = convert_from_path(
                pdf_path,
                dpi=300,  # 高分辨率
                fmt='jpeg',
                thread_count=4
            )
            
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            for i, image in enumerate(images):
                # 图像预处理
                processed_image = self._preprocess_image(image)
                
                # 保存图像
                image_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.jpg")
                processed_image.save(image_path, 'JPEG', quality=95)
                image_paths.append(image_path)
                
                self.logger.info(f"Converted page {i+1} to {image_path}")
            
            return image_paths
            
        except Exception as e:
            self.logger.error(f"PDF conversion failed: {str(e)}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """图像预处理"""
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 图像增强
        import cv2
        img_array = np.array(image)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        # 锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return Image.fromarray(sharpened)
    
    def extract_text_with_positions(self, image_path: str) -> List[Dict[str, Any]]:
        """使用EasyOCR提取文本和位置信息"""
        try:
            # EasyOCR返回格式: [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], text, confidence)]
            results = self.ocr.readtext(image_path)
            
            extracted_data = []
            for bbox, text, confidence in results:
                if confidence > 0.5:  # 过滤低置信度结果
                    # 转换bbox格式为[x1, y1, x2, y2]
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    normalized_bbox = [
                        min(x_coords), min(y_coords),
                        max(x_coords), max(y_coords)
                    ]
                    
                    extracted_data.append({
                        'text': text.strip(),
                        'bbox': normalized_bbox,
                        'confidence': confidence
                    })
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"OCR提取失败: {e}")
            return []
    
    def _normalize_bbox(self, bbox: List[List[float]]) -> List[float]:
        """标准化边界框格式为 [x1, y1, x2, y2]"""
        if len(bbox) == 4 and all(len(point) == 2 for point in bbox):
            # 提取所有x和y坐标
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            # 返回最小和最大坐标
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        else:
            # 如果格式不正确，返回默认值
            return [0, 0, 100, 20]
    
    def load_label_data(self, label_file: str) -> Dict[str, str]:
        """加载标签数据"""
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load label data: {str(e)}")
            return {}
    
    def _smart_label_matching(self, text: str, label_data: Dict[str, str]) -> str:
        """智能标签匹配策略"""
        text_clean = text.strip().lower()
        
        # 发票号码匹配
        if any(keyword in text_clean for keyword in ['发票号', 'invoice', '号码']):
            return "B-InvoiceNo"
        
        # 日期匹配 - 支持多种格式
        date_patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{4}\.\d{1,2}\.\d{1,2}'
        ]
        if any(re.search(pattern, text) for pattern in date_patterns):
            return "B-InvoiceDate"
        
        # 金额匹配 - 改进的数字识别
        amount_patterns = [
            r'[¥$€£]?\s*\d+[,.]?\d*\.?\d*',
            r'\d+[,.]\d+',
            r'[一二三四五六七八九十百千万亿]+元'
        ]
        if any(re.search(pattern, text) for pattern in amount_patterns):
            # 根据上下文判断具体金额类型
            if '税' in text_clean:
                return "B-Tax"
            elif '不含税' in text_clean or 'excluding' in text_clean:
                return "B-AmountwithoutTax"
            else:
                return "B-AmountwithTax"
        
        # 货币匹配
        currencies = ['rmb', 'usd', 'eur', 'jpy', '人民币', '美元', '欧元', '日元']
        if any(curr in text_clean for curr in currencies):
            return "B-Currency"
        
        return "O"
    
    def create_training_annotations(self, image_paths: List[str], label_data: Dict[str, str]) -> List[Dict]:
        """创建训练标注数据"""
        annotations = []
        
        for image_path in image_paths:
            # 提取OCR文本和位置
            ocr_results = self.extract_text_with_positions(image_path)
            
            for ocr_item in ocr_results:
                text = ocr_item['text']
                bbox = ocr_item['bbox']
                confidence = ocr_item['confidence']
                
                # 智能标签匹配
                label = self._smart_label_matching(text, label_data)
                
                # 精确匹配检查
                exact_match_label = self._find_exact_match(text, label_data)
                if exact_match_label:
                    label = exact_match_label
                
                annotation = {
                    'image_path': image_path,
                    'text': text,
                    'bbox': bbox,
                    'label': label,
                    'confidence': confidence
                }
                
                annotations.append(annotation)
        
        self.logger.info(f"Created {len(annotations)} training annotations")
        return annotations
    
    def _find_exact_match(self, text: str, label_data: Dict[str, str]) -> str:
        """查找精确匹配的标签"""
        text_clean = text.strip()
        
        # 直接匹配标签数据中的值
        for field, value in label_data.items():
            if isinstance(value, str) and text_clean == value.strip():
                # 转换为BIO格式
                field_mapping = {
                    "InvoiceNo": "B-InvoiceNo",
                    "InvoiceDate": "B-InvoiceDate",
                    "Currency": "B-Currency",
                    "Amount with Tax": "B-AmountwithTax",
                    "Amount without Tax": "B-AmountwithoutTax",
                    "Tax": "B-Tax"
                }
                return field_mapping.get(field, "O")
        
        return None
    
    def save_annotations(self, annotations: List[Dict], output_file: str):
        """保存标注数据"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved {len(annotations)} annotations to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save annotations: {str(e)}")
            raise
    
    def split_data(self, annotations: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """分割训练和验证数据"""
        import random
        random.shuffle(annotations)
        
        split_idx = int(len(annotations) * train_ratio)
        train_data = annotations[:split_idx]
        val_data = annotations[split_idx:]
        
        self.logger.info(f"Split data: {len(train_data)} training, {len(val_data)} validation")
        return train_data, val_data
    
    def validate_annotations(self, annotations: List[Dict]) -> Dict[str, int]:
        """验证标注数据质量"""
        stats = {
            'total': len(annotations),
            'labeled': 0,
            'unlabeled': 0,
            'low_confidence': 0
        }
        
        label_counts = {}
        
        for ann in annotations:
            if ann['label'] != 'O':
                stats['labeled'] += 1
                label_counts[ann['label']] = label_counts.get(ann['label'], 0) + 1
            else:
                stats['unlabeled'] += 1
            
            if ann['confidence'] < 0.8:
                stats['low_confidence'] += 1
        
        stats['label_distribution'] = label_counts
        
        self.logger.info(f"Annotation validation: {stats}")
        return stats