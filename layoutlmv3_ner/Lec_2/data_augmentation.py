import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

class InvoiceDataAugmentation:
    def __init__(self):
        self.augmentation_methods = [
            self.rotate_image,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_noise,
            self.blur_image
        ]
    
    def rotate_image(self, image, max_angle=5):
        """轻微旋转图像"""
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def adjust_brightness(self, image, factor_range=(0.8, 1.2)):
        """调整亮度"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        factor = random.uniform(*factor_range)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def adjust_contrast(self, image, factor_range=(0.8, 1.2)):
        """调整对比度"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        factor = random.uniform(*factor_range)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def add_noise(self, image, noise_factor=0.1):
        """添加噪声"""
        noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    def blur_image(self, image, kernel_size=3):
        """轻微模糊"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def augment_image(self, image, num_augmentations=2):
        """随机应用多种增强方法"""
        augmented_images = [image]  # 包含原图
        
        for _ in range(num_augmentations):
            current_image = image.copy()
            
            # 随机选择增强方法
            methods = random.sample(self.augmentation_methods, 
                                  random.randint(1, 3))
            
            for method in methods:
                current_image = method(current_image)
            
            augmented_images.append(current_image)
        
        return augmented_images