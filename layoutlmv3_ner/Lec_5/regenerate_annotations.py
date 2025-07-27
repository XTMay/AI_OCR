#!/usr/bin/env python3
"""
Script to regenerate annotations with proper bbox normalization
"""
import sys
import json
from src.data_processor import InvoiceDataProcessor

def main():
    # Initialize processor
    processor = InvoiceDataProcessor("/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data")
    
    # Test with one PDF file first
    pdf_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/raw/測試股份有限公司_1.pdf"
    label_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/validation/label_1.json"
    
    # Load label data
    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    print(f"📄 处理测试文件: {pdf_path}")
    print(f"🏷️ 标签数据: {label_data}")
    
    # Convert PDF to image
    image_paths = processor.pdf_to_images(pdf_path)
    if not image_paths:
        print("❌ PDF转换失败")
        return
    
    print(f"🖼️ 转换得到图像: {image_paths[0]}")
    
    # Create annotations with fixed normalization
    annotations = processor.create_training_annotations([image_paths[0]], label_data)
    
    print(f"\n📊 生成了 {len(annotations)} 个标注:")
    for i, ann in enumerate(annotations[:5]):  # Show first 5
        print(f"  {i+1}. '{ann['text']}' -> {ann['label']} @ {ann['bbox']}")
    
    # Check if coordinates are properly normalized
    all_valid = True
    for ann in annotations:
        bbox = ann['bbox']
        if not all(isinstance(coord, int) and 0 <= coord <= 1000 for coord in bbox):
            print(f"❌ 发现无效bbox: {bbox} for text '{ann['text']}'")
            all_valid = False
    
    if all_valid:
        print("\n✅ 所有bbox坐标都在0-1000范围内！")
    else:
        print("\n❌ 仍有bbox坐标超出范围")
    
    # Save test annotations
    with open("test_annotations.json", 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试标注已保存到: test_annotations.json")

if __name__ == '__main__':
    main()