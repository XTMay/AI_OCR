import cv2
import numpy as np
from typing import List, Dict, Tuple

class AnnotationQualityControl:
    """标注质量控制"""
    
    def __init__(self):
        self.quality_thresholds = {
            'bbox_area_min': 100,  # 最小边界框面积
            'bbox_aspect_ratio_max': 10,  # 最大宽高比
            'text_length_min': 1,  # 最小文本长度
            'confidence_min': 0.7  # 最小置信度
        }
    
    def validate_annotation(self, annotation: Dict) -> Tuple[bool, List[str]]:
        """验证单个标注"""
        errors = []
        
        # 检查必要字段
        required_fields = ['image_path', 'entities']
        for field in required_fields:
            if field not in annotation:
                errors.append(f"缺少必要字段: {field}")
        
        if errors:
            return False, errors
        
        # 检查实体标注
        for i, entity in enumerate(annotation['entities']):
            entity_errors = self._validate_entity(entity, i)
            errors.extend(entity_errors)
        
        return len(errors) == 0, errors
    
    def _validate_entity(self, entity: Dict, index: int) -> List[str]:
        """验证单个实体"""
        errors = []
        prefix = f"实体{index}: "
        
        # 检查边界框
        if 'bbox' not in entity:
            errors.append(f"{prefix}缺少边界框")
        else:
            bbox_errors = self._validate_bbox(entity['bbox'])
            errors.extend([f"{prefix}{err}" for err in bbox_errors])
        
        # 检查文本
        if 'text' not in entity or not entity['text'].strip():
            errors.append(f"{prefix}文本为空")
        elif len(entity['text'].strip()) < self.quality_thresholds['text_length_min']:
            errors.append(f"{prefix}文本过短")
        
        # 检查标签
        if 'label' not in entity:
            errors.append(f"{prefix}缺少标签")
        elif not self._is_valid_label(entity['label']):
            errors.append(f"{prefix}无效标签: {entity['label']}")
        
        return errors
    
    def _validate_bbox(self, bbox: List[int]) -> List[str]:
        """验证边界框"""
        errors = []
        
        if len(bbox) != 4:
            errors.append("边界框格式错误")
            return errors
        
        x_min, y_min, x_max, y_max = bbox
        
        # 检查坐标有效性
        if x_min >= x_max or y_min >= y_max:
            errors.append("边界框坐标无效")
        
        # 检查面积
        area = (x_max - x_min) * (y_max - y_min)
        if area < self.quality_thresholds['bbox_area_min']:
            errors.append(f"边界框面积过小: {area}")
        
        # 检查宽高比
        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = max(width/height, height/width)
        if aspect_ratio > self.quality_thresholds['bbox_aspect_ratio_max']:
            errors.append(f"边界框宽高比异常: {aspect_ratio:.2f}")
        
        return errors
    
    def _is_valid_label(self, label: str) -> bool:
        """检查标签有效性"""
        valid_labels = {
            'B-InvoiceNo', 'I-InvoiceNo',
            'B-InvoiceDate', 'I-InvoiceDate',
            'B-Currency', 'I-Currency',
            'B-AmountWithTax', 'I-AmountWithTax',
            'B-AmountWithoutTax', 'I-AmountWithoutTax',
            'B-Tax', 'I-Tax',
            'O'
        }
        return label in valid_labels
    
    def generate_quality_report(self, annotations: List[Dict]) -> Dict:
        """生成质量报告"""
        total_annotations = len(annotations)
        valid_annotations = 0
        error_summary = {}
        
        for annotation in annotations:
            is_valid, errors = self.validate_annotation(annotation)
            if is_valid:
                valid_annotations += 1
            else:
                for error in errors:
                    error_type = error.split(':')[0] if ':' in error else error
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        return {
            'total_annotations': total_annotations,
            'valid_annotations': valid_annotations,
            'invalid_annotations': total_annotations - valid_annotations,
            'quality_rate': valid_annotations / total_annotations if total_annotations > 0 else 0,
            'error_summary': error_summary
        }