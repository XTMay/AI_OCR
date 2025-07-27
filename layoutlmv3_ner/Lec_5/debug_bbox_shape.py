#!/usr/bin/env python3
"""
Debug bbox tensor shape issue
"""
import json
import torch
from src.layoutlm_trainer import LayoutLMv3Trainer, InvoiceLayoutDataset
from torch.utils.data import DataLoader

def debug_bbox_shape():
    """Debug the bbox tensor shape and values"""
    
    # Initialize trainer and dataset
    trainer = LayoutLMv3Trainer()
    train_dataset = InvoiceLayoutDataset("data/training/demo_train.json", trainer.tokenizer)
    
    print(f"📊 数据集大小: {len(train_dataset)}")
    
    # Check individual samples
    print(f"\n🔍 检查单个样本:")
    for i in range(3):
        sample = train_dataset[i]
        bbox = sample['bbox']
        print(f"样本 {i}:")
        print(f"  bbox形状: {bbox.shape}")
        print(f"  bbox类型: {bbox.dtype}")
        print(f"  bbox范围: [{bbox.min().item():.0f}, {bbox.max().item():.0f}]")
        
        # Check for values > 1000
        if bbox.max() > 1000:
            problematic_values = bbox[bbox > 1000]
            print(f"  ❌ 超出1000的值: {problematic_values}")
        
        # Show first few bbox values
        print(f"  前几个bbox值: {bbox.flatten()[:10]}")
    
    # Check batched data
    print(f"\n🔍 检查批处理数据:")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    for batch_idx, batch in enumerate(train_loader):
        bbox_batch = batch['bbox']
        print(f"批次 {batch_idx}:")
        print(f"  bbox形状: {bbox_batch.shape}")
        print(f"  bbox类型: {bbox_batch.dtype}")
        print(f"  bbox范围: [{bbox_batch.min().item():.0f}, {bbox_batch.max().item():.0f}]")
        
        # Check for values > 1000
        if bbox_batch.max() > 1000:
            problematic_mask = bbox_batch > 1000
            problematic_count = problematic_mask.sum().item()
            problematic_values = bbox_batch[problematic_mask]
            print(f"  ❌ 超出1000的值数量: {problematic_count}")
            print(f"  ❌ 具体值: {problematic_values[:20]}")  # Show first 20
        else:
            print(f"  ✅ 所有值都在范围内")
        
        if batch_idx >= 2:  # Check first 3 batches
            break
    
    # Try to find the exact problematic sample
    print(f"\n🔍 寻找问题样本:")
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        bbox = sample['bbox']
        
        if bbox.max() > 1000:
            print(f"❌ 发现问题样本 {i}: bbox范围 [{bbox.min().item():.0f}, {bbox.max().item():.0f}]")
            
            # Get the original data
            with open("data/training/demo_train.json", 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            # Find the corresponding data sample
            sample_count = 0
            for item in train_data:
                if sample_count == i:
                    print(f"原始数据:")
                    print(f"  图像: {item['image_path']}")
                    for j, entity in enumerate(item['entities'][:3]):
                        print(f"  实体 {j}: '{entity['text']}' bbox={entity['bbox']}")
                    break
                sample_count += len(item['entities'])
                if sample_count > i:
                    break
        
        if i >= 50:  # Don't check all samples, just first 50
            break

if __name__ == '__main__':
    debug_bbox_shape()