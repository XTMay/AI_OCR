import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Tuple, List, Dict

class InvoiceDataAugmentation:
    """发票数据增强"""
    
    def __init__(self):
        self.augmentation_config = {
            'rotation_range': (-5, 5),  # 旋转角度范围
            'brightness_range': (0.8, 1.2),  # 亮度范围
            'contrast_range': (0.8, 1.2),  # 对比度范围
            'noise_intensity': 0.02,  # 噪声强度
            'blur_kernel_size': (1, 3),  # 模糊核大小
            'perspective_strength': 0.1  # 透视变换强度
        }
    
    def augment_image_and_annotations(self, image_path: str, annotations: List[Dict]) -> List[Tuple[np.ndarray, List[Dict]]]:
        """图像和标注同步增强"""
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        augmented_samples = []
        
        # 原始样本
        augmented_samples.append((image.copy(), annotations.copy()))
        
        # 应用各种增强
        augmentation_methods = [
            self._apply_rotation,
            self._apply_brightness_contrast,
            self._apply_noise,
            self._apply_blur,
            self._apply_perspective_transform
        ]
        
        for method in augmentation_methods:
            try:
                aug_image, aug_annotations = method(image.copy(), annotations.copy())
                if aug_image is not None and aug_annotations:
                    augmented_samples.append((aug_image, aug_annotations))
            except Exception as e:
                print(f"增强方法 {method.__name__} 失败: {e}")
                continue
        
        return augmented_samples
    
    def _apply_rotation(self, image: np.ndarray, annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """应用旋转增强"""
        angle = random.uniform(*self.augmentation_config['rotation_range'])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转图像
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # 旋转边界框
        rotated_annotations = []
        for ann in annotations:
            rotated_bbox = self._rotate_bbox(ann['bbox'], rotation_matrix, (w, h))
            if rotated_bbox:
                rotated_ann = ann.copy()
                rotated_ann['bbox'] = rotated_bbox
                rotated_annotations.append(rotated_ann)
        
        return rotated_image, rotated_annotations
    
    def _rotate_bbox(self, bbox: List[int], rotation_matrix: np.ndarray, image_size: Tuple[int, int]) -> List[int]:
        """旋转边界框"""
        x_min, y_min, x_max, y_max = bbox
        w, h = image_size
        
        # 边界框的四个角点
        corners = np.array([
            [x_min, y_min, 1],
            [x_max, y_min, 1],
            [x_max, y_max, 1],
            [x_min, y_max, 1]
        ]).T
        
        # 应用旋转
        rotated_corners = rotation_matrix @ corners
        
        # 计算新的边界框
        x_coords = rotated_corners[0, :]
        y_coords = rotated_corners[1, :]
        
        new_x_min = max(0, int(np.min(x_coords)))
        new_y_min = max(0, int(np.min(y_coords)))
        new_x_max = min(w, int(np.max(x_coords)))
        new_y_max = min(h, int(np.max(y_coords)))
        
        # 检查边界框有效性
        if new_x_max <= new_x_min or new_y_max <= new_y_min:
            return None
        
        return [new_x_min, new_y_min, new_x_max, new_y_max]
    
    def _apply_brightness_contrast(self, image: np.ndarray, annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """应用亮度和对比度增强"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 调整亮度
        brightness_factor = random.uniform(*self.augmentation_config['brightness_range'])
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        # 调整对比度
        contrast_factor = random.uniform(*self.augmentation_config['contrast_range'])
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        # 转换回OpenCV格式
        enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced_image, annotations
    
    def _apply_noise(self, image: np.ndarray, annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """应用噪声"""
        noise = np.random.normal(0, self.augmentation_config['noise_intensity'] * 255, image.shape)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return noisy_image, annotations
    
    def _apply_blur(self, image: np.ndarray, annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """应用模糊"""
        kernel_size = random.randint(*self.augmentation_config['blur_kernel_size'])
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return blurred_image, annotations
    
    def _apply_perspective_transform(self, image: np.ndarray, annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """应用透视变换"""
        h, w = image.shape[:2]
        strength = self.augmentation_config['perspective_strength']
        
        # 定义源点和目标点
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # 随机扰动角点
        dst_points = src_points.copy()
        for i in range(4):
            dst_points[i][0] += random.uniform(-strength * w, strength * w)
            dst_points[i][1] += random.uniform(-strength * h, strength * h)
        
        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        transformed_image = cv2.warpPerspective(image, perspective_matrix, (w, h))
        
        # 变换边界框
        transformed_annotations = []
        for ann in annotations:
            transformed_bbox = self._transform_bbox_perspective(ann['bbox'], perspective_matrix)
            if transformed_bbox:
                transformed_ann = ann.copy()
                transformed_ann['bbox'] = transformed_bbox
                transformed_annotations.append(transformed_ann)
        
        return transformed_image, transformed_annotations
    
    def _transform_bbox_perspective(self, bbox: List[int], perspective_matrix: np.ndarray) -> List[int]:
        """透视变换边界框"""
        x_min, y_min, x_max, y_max = bbox
        
        # 边界框的四个角点
        corners = np.array([
            [[x_min, y_min]],
            [[x_max, y_min]],
            [[x_max, y_max]],
            [[x_min, y_max]]
        ], dtype=np.float32)
        
        # 应用透视变换
        transformed_corners = cv2.perspectiveTransform(corners, perspective_matrix)
        
        # 计算新的边界框
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        new_x_min = max(0, int(np.min(x_coords)))
        new_y_min = max(0, int(np.min(y_coords)))
        new_x_max = int(np.max(x_coords))
        new_y_max = int(np.max(y_coords))
        
        # 检查边界框有效性
        if new_x_max <= new_x_min or new_y_max <= new_y_min:
            return None
        
        return [new_x_min, new_y_min, new_x_max, new_y_max]