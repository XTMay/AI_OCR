#!/usr/bin/env python3
"""
Debug script to check bbox normalization
"""
import json
from PIL import Image

def debug_bbox_issue():
    # Check the actual image dimensions
    image_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg"
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"🖼️ 图像尺寸: {width} x {height}")
    except Exception as e:
        print(f"❌ 无法读取图像: {e}")
        return
    
    # Check current annotation data
    annotation_file = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/all_annotations_20250724_201047.json"
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"\n📋 检查前3个标注的bbox:")
        for i, ann in enumerate(annotations[:3]):
            text = ann['text']
            bbox = ann['bbox']
            print(f"\n标注 {i+1}: '{text}'")
            print(f"  原始bbox: {bbox}")
            
            # Manual normalization test
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                norm_x1 = int((x1 / width) * 1000)
                norm_y1 = int((y1 / height) * 1000)
                norm_x2 = int((x2 / width) * 1000)
                norm_y2 = int((y2 / height) * 1000)
                
                print(f"  应该归一化为: [{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]")
                
                # Check if already normalized
                if all(coord <= 1000 for coord in bbox):
                    if width > 1000 or height > 1000:
                        print("  🤔 坐标看起来已经归一化了，但图像尺寸大于1000")
                    else:
                        print("  ✅ 坐标可能已经正确归一化")
                else:
                    print("  ❌ 坐标需要归一化")
            
    except Exception as e:
        print(f"❌ 读取标注失败: {e}")

if __name__ == '__main__':
    debug_bbox_issue()