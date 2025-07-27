#!/usr/bin/env python3
"""
Debug the LayoutLM tokenizer bbox processing
"""
import json
from transformers import LayoutLMv3Tokenizer

def debug_tokenizer_bbox():
    """Debug what happens to bbox coordinates in the tokenizer"""
    
    # Load training data
    with open("data/training/demo_train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Initialize tokenizer
    tokenizer = LayoutLMv3Tokenizer.from_pretrained('microsoft/layoutlmv3-base')
    
    # Get first sample
    sample = train_data[0]
    print(f"📄 测试样本: {sample['image_path']}")
    print(f"📊 实体数量: {len(sample['entities'])}")
    
    # Process like InvoiceLayoutDataset does
    words = []
    boxes = []
    labels = []
    
    label_to_id = {
        'O': 0,
        'B-InvoiceNo': 1, 'I-InvoiceNo': 2,
        'B-InvoiceDate': 3, 'I-InvoiceDate': 4,
        'B-Currency': 5, 'I-Currency': 6,
        'B-AmountwithTax': 7, 'I-AmountwithTax': 8,
        'B-AmountwithoutTax': 9, 'I-AmountwithoutTax': 10,
        'B-Tax': 11, 'I-Tax': 12
    }
    
    print(f"\n🔍 处理前3个实体:")
    for i, item in enumerate(sample['entities'][:3]):
        print(f"\n实体 {i+1}:")
        print(f"  文本: '{item['text']}'")
        print(f"  标签: {item['label']}")
        print(f"  bbox: {item['bbox']}")
        
        # Check bbox values
        bbox = item['bbox']
        if any(coord > 1000 for coord in bbox):
            print(f"  ❌ bbox坐标超出1000: max={max(bbox)}")
        else:
            print(f"  ✅ bbox坐标在范围内: max={max(bbox)}")
        
        # Tokenize like the dataset does
        word_tokens = tokenizer.tokenize(item['text'])
        words.extend(word_tokens)
        
        for j, token in enumerate(word_tokens):
            boxes.append(item['bbox'])
            if j == 0:
                labels.append(label_to_id[item['label']])
            else:
                i_label = item['label'].replace('B-', 'I-')
                labels.append(label_to_id.get(i_label, label_to_id['O']))
    
    print(f"\n📝 准备传给tokenizer:")
    print(f"  words: {words}")
    print(f"  boxes: {boxes}")
    print(f"  labels: {labels}")
    
    try:
        # Try tokenizer encoding
        encoding = tokenizer(
            words,
            boxes=boxes,
            word_labels=labels,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        print(f"\n✅ Tokenizer编码成功!")
        bbox_tensor = encoding['bbox']
        print(f"📊 输出bbox形状: {bbox_tensor.shape}")
        print(f"📊 bbox值范围: [{bbox_tensor.min().item():.0f}, {bbox_tensor.max().item():.0f}]")
        
        if bbox_tensor.max().item() > 1000:
            print(f"❌ Tokenizer输出的bbox超出1000: {bbox_tensor.max().item()}")
        else:
            print(f"✅ Tokenizer输出的bbox在正确范围内")
            
    except Exception as e:
        print(f"❌ Tokenizer编码失败: {e}")
        if "bbox" in str(e) and "0-1000" in str(e):
            print("这就是导致训练失败的错误！")

if __name__ == '__main__':
    debug_tokenizer_bbox()