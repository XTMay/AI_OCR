import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class AdvancedDataAugmentation:
    def __init__(self):
        # 定义高级数据增强管道
        self.transform = A.Compose([
            # 几何变换
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=3,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=0.7
            ),
            
            # 透视变换
            A.Perspective(
                scale=(0.02, 0.05),
                keep_size=True,
                p=0.3
            ),
            
            # 光照变化
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=1.0
                ),
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=1.0
                )
            ], p=0.8),
            
            # 噪声和模糊
            A.OneOf([
                A.GaussNoise(
                    var_limit=(10, 50),
                    p=1.0
                ),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    p=1.0
                )
            ], p=0.5),
            
            # 模糊效果
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=1.0)
            ], p=0.3),
            
            # 压缩伪影
            A.ImageCompression(
                quality_lower=75,
                quality_upper=100,
                p=0.3
            )
        ])
    
    def augment_invoice_image(self, image, bboxes=None):
        """针对发票图像的专门增强"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # 应用增强
        if bboxes is not None:
            # 保护边界框的增强
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=['class_labels']
                )
            )
            return transformed['image'], transformed['bboxes']
        else:
            transformed = self.transform(image=image)
            return transformed['image']
    
    def synthetic_data_generation(self, template_image, text_data):
        """合成数据生成"""
        # 基于模板生成合成发票数据
        synthetic_images = []
        
        for text_info in text_data:
            # 在模板上渲染新的文本
            synthetic_img = self.render_text_on_template(
                template_image, 
                text_info
            )
            synthetic_images.append(synthetic_img)
        
        return synthetic_images