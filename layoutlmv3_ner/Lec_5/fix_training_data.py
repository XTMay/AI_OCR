#!/usr/bin/env python3
"""
Fix the training data by applying proper bbox normalization
"""
import json
import os

def fix_training_data():
    """Apply bbox normalization to existing training data"""
    
    # Image dimensions from the log (2481x3508)
    img_width = 2481
    img_height = 3508
    
    # Read the existing training data
    train_file = "data/training/demo_train.json"
    if not os.path.exists(train_file):
        print("❌ 训练文件不存在")
        return False
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"🔧 修复 {len(train_data)} 个训练样本的bbox坐标")
    
    fixed_data = []
    total_entities = 0
    fixed_entities = 0
    
    for item in train_data:
        fixed_item = {
            'image_path': item['image_path'],
            'entities': []
        }
        
        for entity in item['entities']:
            bbox = entity['bbox']
            total_entities += 1
            
            # Check if bbox needs normalization (all coordinates should be in 0-1000 range after normalization)
            # If coordinates look like pixel values (large numbers relative to 1000), normalize them
            if len(bbox) == 4 and (max(bbox) > 500 or (max(bbox) < 1000 and max(bbox) > 100)):
                # Apply normalization
                x1, y1, x2, y2 = bbox
                
                norm_x1 = max(0, min(1000, int((x1 / img_width) * 1000)))
                norm_y1 = max(0, min(1000, int((y1 / img_height) * 1000)))
                norm_x2 = max(0, min(1000, int((x2 / img_width) * 1000)))
                norm_y2 = max(0, min(1000, int((y2 / img_height) * 1000)))
                
                # Ensure x2 > x1, y2 > y1
                if norm_x2 <= norm_x1:
                    norm_x2 = min(1000, norm_x1 + 1)
                if norm_y2 <= norm_y1:
                    norm_y2 = min(1000, norm_y1 + 1)
                
                normalized_bbox = [norm_x1, norm_y1, norm_x2, norm_y2]
                
                print(f"  📝 '{entity['text'][:20]}': {bbox} -> {normalized_bbox}")
                
                fixed_entity = {
                    'text': entity['text'],
                    'bbox': normalized_bbox,
                    'label': entity['label']
                }
                fixed_entities += 1
            else:
                # Keep original if already normalized
                fixed_entity = entity.copy()
            
            fixed_item['entities'].append(fixed_entity)
        
        fixed_data.append(fixed_item)
    
    # Save the fixed data
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 修复完成:")
    print(f"  总实体数: {total_entities}")
    print(f"  修复实体数: {fixed_entities}")
    print(f"  保存到: {train_file}")
    
    # Validate the fixed data
    with open(train_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    all_valid = True
    for item in validation_data:
        for entity in item['entities']:
            bbox = entity['bbox']
            if not all(isinstance(coord, int) and 0 <= coord <= 1000 for coord in bbox):
                print(f"❌ 验证失败: {bbox}")
                all_valid = False
                break
        if not all_valid:
            break
    
    if all_valid:
        print("✅ 所有bbox坐标都在0-1000范围内！")
        return True
    else:
        print("❌ 仍有无效的bbox坐标")
        return False

if __name__ == '__main__':
    success = fix_training_data()
    if success:
        print("\n🎉 训练数据修复成功！现在可以运行LayoutLM训练了。")
    else:
        print("\n❌ 训练数据修复失败")