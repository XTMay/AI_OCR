import json
import requests
from label_studio_sdk import Client
from typing import List, Dict, Any

class LabelStudioIntegration:
    """Label Studio集成工具"""
    
    def __init__(self, url: str, api_key: str):
        self.client = Client(url=url, api_key=api_key)
        self.url = url
        self.api_key = api_key
    
    def create_project(self, title: str, config_path: str) -> int:
        """创建标注项目"""
        with open(config_path, 'r', encoding='utf-8') as f:
            label_config = f.read()
        
        project = self.client.start_project(
            title=title,
            label_config=label_config
        )
        return project.id
    
    def upload_images(self, project_id: int, image_paths: List[str]):
        """上传图像到项目"""
        project = self.client.get_project(project_id)
        
        for image_path in image_paths:
            task = {
                "data": {"image": f"/data/local-files/?d={image_path}"}
            }
            project.import_tasks([task])
    
    def export_annotations(self, project_id: int, export_format: str = 'JSON') -> List[Dict]:
        """导出标注数据"""
        project = self.client.get_project(project_id)
        annotations = project.export_tasks(export_type=export_format)
        return annotations
    
    def convert_to_layoutlm_format(self, annotations: List[Dict]) -> List[Dict]:
        """转换为LayoutLM训练格式"""
        converted_data = []
        
        for annotation in annotations:
            if not annotation.get('annotations'):
                continue
                
            image_path = annotation['data']['image']
            entities = []
            
            for ann in annotation['annotations']:
                for result in ann.get('result', []):
                    if result['type'] == 'rectanglelabels':
                        # 提取边界框和标签
                        bbox = self._convert_bbox(result['value'])
                        label = result['value']['rectanglelabels'][0]
                        text = result.get('value', {}).get('text', '')
                        
                        entities.append({
                            'text': text,
                            'bbox': bbox,
                            'label': f'B-{label}'
                        })
            
            if entities:
                converted_data.append({
                    'image_path': image_path,
                    'entities': entities
                })
        
        return converted_data
    
    def _convert_bbox(self, value: Dict) -> List[int]:
        """转换边界框格式"""
        # Label Studio使用百分比坐标，需要转换为像素坐标
        x = value['x'] / 100
        y = value['y'] / 100
        width = value['width'] / 100
        height = value['height'] / 100
        
        # 这里需要图像的实际尺寸来计算像素坐标
        # 假设图像尺寸为1000x1000（实际使用时需要获取真实尺寸）
        img_width, img_height = 1000, 1000
        
        x_min = int(x * img_width)
        y_min = int(y * img_height)
        x_max = int((x + width) * img_width)
        y_max = int((y + height) * img_height)
        
        return [x_min, y_min, x_max, y_max]