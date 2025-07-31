import re
import random
import numpy as np
from typing import List, Dict, Tuple
from copy import deepcopy

class InvoiceDataAugmentation:
    """发票数据增强"""
    
    def __init__(self, position_noise_ratio: float = 0.05, rotation_angle_range: Tuple[float, float] = (-2, 2)):
        """
        初始化数据增强器
        
        Args:
            position_noise_ratio: 位置噪声比例
            rotation_angle_range: 旋转角度范围
        """
        self.position_noise_ratio = position_noise_ratio
        self.rotation_angle_range = rotation_angle_range
    
    def augment_invoice_data(self, annotations: List[Dict]) -> List[Dict]:
        """多种数据增强技术"""
        augmented_data = []
        
        for ann in annotations:
            # 原始数据
            augmented_data.append(ann)
            
            # 1. 文本变换
            text_variants = self._generate_text_variants(ann['text'])
            for variant in text_variants:
                aug_ann = ann.copy()
                aug_ann['text'] = variant
                augmented_data.append(aug_ann)
            
            # 2. 位置扰动
            position_variants = self._generate_position_variants(ann)
            augmented_data.extend(position_variants)
        
        return augmented_data
    
    def _generate_text_variants(self, text: str) -> List[str]:
        """生成文本变体"""
        variants = []
        
        # 大小写变换
        variants.append(text.upper())
        variants.append(text.lower())
        
        # 数字格式变换
        if re.search(r'\d', text):
            # 添加千分位分隔符
            variants.append(re.sub(r'(\d)(?=(\d{3})+(?!\d))', r'\1,', text))
        
        # 日期格式变换
        date_patterns = [
            (r'(\d{4})-(\d{2})-(\d{2})', r'\1/\2/\3'),
            (r'(\d{4})/(\d{2})/(\d{2})', r'\1.\2.\3'),
        ]
        
        for pattern, replacement in date_patterns:
            variant = re.sub(pattern, replacement, text)
            if variant != text:
                variants.append(variant)
        
        return variants[:3]  # 限制变体数量
    
    def _generate_position_variants(self, annotation: Dict) -> List[Dict]:
        """生成位置变体"""
        variants = []
        
        # 获取原始边界框
        bbox = annotation.get('bbox', [])
        if len(bbox) != 4:
            return variants
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 1. 轻微位置偏移
        for _ in range(2):
            noise_x = random.uniform(-width * self.position_noise_ratio, width * self.position_noise_ratio)
            noise_y = random.uniform(-height * self.position_noise_ratio, height * self.position_noise_ratio)
            
            new_bbox = [
                max(0, x1 + noise_x),
                max(0, y1 + noise_y),
                x2 + noise_x,
                y2 + noise_y
            ]
            
            variant = deepcopy(annotation)
            variant['bbox'] = new_bbox
            variants.append(variant)
        
        # 2. 轻微尺寸变化
        scale_factors = [0.95, 1.05]
        for scale in scale_factors:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            new_width = width * scale
            new_height = height * scale
            
            new_bbox = [
                center_x - new_width / 2,
                center_y - new_height / 2,
                center_x + new_width / 2,
                center_y + new_height / 2
            ]
            
            variant = deepcopy(annotation)
            variant['bbox'] = new_bbox
            variants.append(variant)
        
        return variants
    
    def augment_with_synthetic_noise(self, annotations: List[Dict], noise_level: float = 0.1) -> List[Dict]:
        """添加合成噪声"""
        augmented = []
        
        for ann in annotations:
            # 原始数据
            augmented.append(ann)
            
            # 添加OCR错误模拟
            if random.random() < noise_level:
                noisy_ann = self._simulate_ocr_errors(ann)
                augmented.append(noisy_ann)
        
        return augmented
    
    def _simulate_ocr_errors(self, annotation: Dict) -> Dict:
        """模拟OCR错误"""
        text = annotation['text']
        
        # 常见OCR错误替换
        ocr_errors = {
            '0': ['O', 'o'],
            '1': ['l', 'I'],
            '5': ['S'],
            '8': ['B'],
            'O': ['0'],
            'l': ['1'],
            'I': ['1'],
            'S': ['5'],
            'B': ['8']
        }
        
        noisy_text = text
        for char in text:
            if char in ocr_errors and random.random() < 0.1:
                replacement = random.choice(ocr_errors[char])
                noisy_text = noisy_text.replace(char, replacement, 1)
        
        noisy_ann = deepcopy(annotation)
        noisy_ann['text'] = noisy_text
        return noisy_ann
    
    def balance_dataset(self, annotations: List[Dict]) -> List[Dict]:
        """平衡数据集"""
        # 统计各标签数量
        label_counts = {}
        for ann in annotations:
            label = ann.get('label', 'O')
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 找到最大数量
        max_count = max(label_counts.values())
        
        # 为少数类生成更多样本
        balanced_data = list(annotations)
        for label, count in label_counts.items():
            if count < max_count:
                # 找到该标签的所有样本
                label_samples = [ann for ann in annotations if ann.get('label') == label]
                
                # 需要增加的样本数
                needed = max_count - count
                
                # 通过重复和变换生成新样本
                for _ in range(needed):
                    base_sample = random.choice(label_samples)
                    augmented_sample = self._create_augmented_sample(base_sample)
                    balanced_data.append(augmented_sample)
        
        return balanced_data
    
    def _create_augmented_sample(self, sample: Dict) -> Dict:
        """创建增强样本"""
        augmented = deepcopy(sample)
        
        # 应用文本变换
        text_variants = self._generate_text_variants(sample['text'])
        if text_variants:
            augmented['text'] = random.choice(text_variants)
        
        # 应用位置变换
        position_variants = self._generate_position_variants(sample)
        if position_variants:
            position_variant = random.choice(position_variants)
            augmented['bbox'] = position_variant['bbox']
        
        return augmented