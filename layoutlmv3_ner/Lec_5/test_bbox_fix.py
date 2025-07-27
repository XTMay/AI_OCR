#!/usr/bin/env python3
"""
Test script to verify bbox coordinate normalization fix
"""
import json
import os

def test_bbox_normalization():
    """Test the bbox coordinate values in the saved annotations"""
    annotation_file = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/all_annotations_20250724_201047.json"
    
    if not os.path.exists(annotation_file):
        print("❌ Annotation file not found!")
        return False
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"🔍 检查 {len(annotations)} 个标注的bbox坐标...")
        
        invalid_count = 0
        valid_count = 0
        
        for i, annotation in enumerate(annotations[:10]):  # Check first 10
            bbox = annotation.get('bbox', [])
            text = annotation.get('text', '')
            
            print(f"\n标注 {i+1}: '{text[:20]}...'")
            print(f"  bbox: {bbox}")
            
            # Check if bbox has 4 coordinates
            if len(bbox) != 4:
                print(f"  ❌ bbox长度错误: {len(bbox)}")
                invalid_count += 1
                continue
            
            # Check if all coordinates are in 0-1000 range
            all_valid = True
            for j, coord in enumerate(bbox):
                if not isinstance(coord, (int, float)) or coord < 0 or coord > 1000:
                    print(f"  ❌ 坐标 {j} 超出范围: {coord}")
                    all_valid = False
            
            if all_valid:
                print(f"  ✅ bbox坐标有效")
                valid_count += 1
            else:
                invalid_count += 1
        
        print(f"\n📊 检查结果:")
        print(f"  ✅ 有效标注: {valid_count}")
        print(f"  ❌ 无效标注: {invalid_count}")
        
        return invalid_count == 0
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def test_demo_train_json():
    """Check the demo_train.json file if it exists"""
    demo_file = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/training/demo_train.json"
    
    if not os.path.exists(demo_file):
        print("📝 demo_train.json 文件不存在，需要重新生成")
        return True
    
    try:
        with open(demo_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"🔍 检查 demo_train.json 中的bbox坐标...")
        
        for i, item in enumerate(data[:3]):  # Check first 3 items
            entities = item.get('entities', [])
            print(f"\n训练样本 {i+1}:")
            for j, entity in enumerate(entities[:3]):  # Check first 3 entities
                bbox = entity.get('bbox', [])
                text = entity.get('text', '')
                print(f"  实体 {j+1}: '{text}' bbox: {bbox}")
                
                # Check coordinate range
                if len(bbox) == 4:
                    if all(isinstance(coord, (int, float)) and 0 <= coord <= 1000 for coord in bbox):
                        print(f"    ✅ 坐标有效")
                    else:
                        print(f"    ❌ 坐标无效")
                        return False
                else:
                    print(f"    ❌ bbox格式错误")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ 检查demo_train.json失败: {e}")
        return False

if __name__ == '__main__':
    print("🧪 开始测试bbox坐标修复...")
    
    # Test annotation files
    print("\n1. 检查标注文件:")
    result1 = test_bbox_normalization()
    
    # Test training data
    print("\n2. 检查训练数据:")
    result2 = test_demo_train_json()
    
    print(f"\n🎯 测试结果:")
    print(f"  标注文件: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"  训练数据: {'✅ 通过' if result2 else '❌ 失败'}")
    
    if result1 and result2:
        print("\n🎉 所有测试通过！bbox坐标修复成功！")
    else:
        print("\n⚠️ 某些测试失败，需要进一步检查")