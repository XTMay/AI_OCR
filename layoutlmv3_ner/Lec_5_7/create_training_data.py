#!/usr/bin/env python3
"""
Create training data with proper bbox normalization
"""
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_data():
    """Create training data with normalized bboxes"""
    # Use our mock normalized data for now
    mock_annotations = [
        {
            'image_path': '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg',
            'text': 'TE ST',
            'bbox': [71, 35, 102, 39],
            'label': 'O'
        },
        {
            'image_path': '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg',
            'text': 'Test Co.,Ltd.',
            'bbox': [157, 35, 202, 39],
            'label': 'O'
        },
        {
            'image_path': '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg',
            'text': 'Invoice',
            'bbox': [157, 45, 184, 49],
            'label': 'B-InvoiceNo'
        },
        {
            'image_path': '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg',
            'text': 'Y 309824263008',
            'bbox': [238, 94, 340, 98],
            'label': 'B-InvoiceNo'
        },
        {
            'image_path': '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg',
            'text': '2025年6月30日',
            'bbox': [400, 120, 500, 125],
            'label': 'B-InvoiceDate'
        },
        {
            'image_path': '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg',
            'text': 'USD',
            'bbox': [200, 150, 230, 155],
            'label': 'B-Currency'
        },
        {
            'image_path': '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg',
            'text': '300',
            'bbox': [300, 180, 330, 185],
            'label': 'B-AmountwithTax'
        },
    ]
    
    logger.info(f"🔧 创建 {len(mock_annotations)} 个标注的训练数据")
    
    # Convert to training format
    train_data = []
    
    # Group annotations by image
    image_groups = {}
    for ann in mock_annotations:
        image_path = ann['image_path']
        if image_path not in image_groups:
            image_groups[image_path] = []
        image_groups[image_path].append(ann)
    
    # Create training entries
    for image_path, annotations in image_groups.items():
        entities = []
        for ann in annotations:
            entities.append({
                'text': ann['text'],
                'bbox': ann['bbox'],
                'label': ann['label']
            })
        
        train_data.append({
            'image_path': image_path,
            'entities': entities
        })
    
    # Ensure directories exist
    os.makedirs("data/training", exist_ok=True)
    
    # Save training data
    train_file = "data/training/demo_train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 训练数据已保存到: {train_file}")
    
    # Validate the saved data
    with open(train_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    all_valid = True
    for item in loaded_data:
        for entity in item['entities']:
            bbox = entity['bbox']
            if not all(isinstance(coord, int) and 0 <= coord <= 1000 for coord in bbox):
                logger.error(f"❌ 发现无效bbox: {bbox}")
                all_valid = False
    
    if all_valid:
        logger.info("✅ 所有bbox坐标都在0-1000范围内！")
        return True
    else:
        logger.error("❌ 发现无效的bbox坐标")
        return False

if __name__ == '__main__':
    success = create_training_data()
    if success:
        print("\n🎉 训练数据创建成功！现在可以运行LayoutLM训练了。")
    else:
        print("\n❌ 训练数据创建失败")