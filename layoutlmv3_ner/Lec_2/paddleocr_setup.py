from paddleocr import PaddleOCR
import cv2
import numpy as np

class AdvancedPaddleOCR:
    def __init__(self):
        # 初始化PaddleOCR，使用中英文模型
        self.ocr = PaddleOCR(
            lang='ch',  # 支持中文
            det_model_dir=None,  # 使用默认检测模型
            rec_model_dir=None,  # 使用默认识别模型
            use_angle_cls=True,  # 启用文字方向分类
            use_gpu=False  # 根据硬件情况调整
        )
    
    def extract_text_with_layout(self, image_path):
        """提取文本及其布局信息"""
        # 读取图像
        image = cv2.imread(image_path)
        
        # OCR识别
        result = self.ocr.ocr(image_path)
        
        # 解析结果
        text_blocks = []
        for line in result[0]:
            bbox = line[0]  # 边界框坐标
            text = line[1][0]  # 识别的文本
            confidence = line[1][1]  # 置信度
            
            # 计算文本块的中心点和尺寸
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            text_block = {
                'text': text,
                'bbox': {
                    'x_min': min(x_coords),
                    'y_min': min(y_coords),
                    'x_max': max(x_coords),
                    'y_max': max(y_coords)
                },
                'confidence': confidence,
                'center_x': sum(x_coords) / 4,
                'center_y': sum(y_coords) / 4
            }
            text_blocks.append(text_block)
            
        return text_blocks
    
    def create_layout_features(self, text_blocks, image_shape):
        """创建布局特征用于LayoutLM"""
        layout_features = []
        
        for block in text_blocks:
            # 归一化坐标 (LayoutLM要求0-1000范围)
            normalized_bbox = [
                int(block['bbox']['x_min'] / image_shape[1] * 1000),
                int(block['bbox']['y_min'] / image_shape[0] * 1000),
                int(block['bbox']['x_max'] / image_shape[1] * 1000),
                int(block['bbox']['y_max'] / image_shape[0] * 1000)
            ]
            
            layout_features.append({
                'text': block['text'],
                'bbox': normalized_bbox,
                'confidence': block['confidence']
            })
            
        return layout_features