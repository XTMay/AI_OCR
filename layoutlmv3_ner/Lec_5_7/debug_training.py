#!/usr/bin/env python3
"""
Debug exactly where the bbox training error occurs
"""
import json
import logging
import traceback
from src.layoutlm_trainer import LayoutLMv3Trainer, InvoiceLayoutDataset

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_training_error():
    """Debug the exact point where bbox training fails"""
    try:
        logger.info("🔧 开始调试训练错误...")
        
        # Initialize trainer
        trainer = LayoutLMv3Trainer()
        logger.info("✅ 训练器初始化成功")
        
        # Create dataset
        train_file = "data/training/demo_train.json"
        train_dataset = InvoiceLayoutDataset(train_file, trainer.tokenizer)
        logger.info(f"✅ 数据集创建成功，大小: {len(train_dataset)}")
        
        # Test multiple samples
        for i in range(min(5, len(train_dataset))):
            try:
                sample = train_dataset[i]
                bbox = sample['bbox']
                logger.info(f"样本 {i}: bbox范围 [{bbox.min().item():.0f}, {bbox.max().item():.0f}]")
                
                if bbox.max().item() > 1000:
                    logger.error(f"❌ 样本 {i} bbox超出范围: {bbox.max().item()}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ 样本 {i} 处理失败: {e}")
                return False
        
        logger.info("✅ 所有样本bbox都在正确范围内")
        
        # Try setup training
        try:
            trainer.setup_training(
                train_dataset, 
                train_dataset,
                "./models/debug_training",
                num_epochs=1
            )
            logger.info("✅ 训练设置成功")
        except Exception as e:
            logger.error(f"❌ 训练设置失败: {e}")
            if "bbox" in str(e) and "0-1000" in str(e):
                logger.error("这是bbox范围错误！")
                traceback.print_exc()
            return False
        
        # Try one training step
        try:
            logger.info("🏃‍♂️ 尝试执行一个训练步骤...")
            # This will likely fail, but we want to see exactly where
            trainer.train()
            
        except Exception as e:
            logger.error(f"❌ 训练步骤失败: {e}")
            if "bbox" in str(e) and "0-1000" in str(e):
                logger.error("发现bbox范围错误的确切位置！")
                traceback.print_exc()
                
                # Try to find the problematic batch
                logger.info("🔍 分析训练数据批次...")
                from torch.utils.data import DataLoader
                
                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
                for batch_idx, batch in enumerate(train_loader):
                    bbox_batch = batch['bbox']
                    logger.info(f"批次 {batch_idx}: bbox范围 [{bbox_batch.min().item():.0f}, {bbox_batch.max().item():.0f}]")
                    
                    if bbox_batch.max().item() > 1000:
                        logger.error(f"❌ 批次 {batch_idx} 包含超出范围的bbox: {bbox_batch.max().item()}")
                        
                        # Find the specific problematic values
                        problematic_mask = bbox_batch > 1000
                        problematic_values = bbox_batch[problematic_mask]
                        logger.error(f"具体的问题值: {problematic_values}")
                        break
                    
                    if batch_idx >= 3:  # Only check first few batches
                        break
                        
            return False
        
        logger.info("🎉 训练步骤成功！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 调试过程失败: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = debug_training_error()
    if success:
        print("\n🎉 没有发现bbox问题！")
    else:
        print("\n🔍 发现了bbox问题的根源")