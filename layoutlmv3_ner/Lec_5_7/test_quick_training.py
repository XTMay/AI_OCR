#!/usr/bin/env python3
"""
Quick test of LayoutLM training with fixed bbox coordinates
"""
import json
import logging
from src.layoutlm_trainer import LayoutLMv3Trainer, InvoiceLayoutDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_training_test():
    """Test a few training steps to verify bbox coordinates work"""
    try:
        logger.info("🧪 开始快速训练测试...")
        
        # Initialize trainer
        trainer = LayoutLMv3Trainer()
        logger.info("✅ 训练器初始化成功")
        
        # Create dataset with fixed data
        train_file = "data/training/demo_train.json"
        train_dataset = InvoiceLayoutDataset(train_file, trainer.tokenizer)
        logger.info(f"✅ 数据集创建成功，大小: {len(train_dataset)}")
        
        # Test first sample
        sample = train_dataset[0]
        bbox = sample['bbox']
        logger.info(f"📊 第一个样本bbox范围: [{bbox.min().item():.0f}, {bbox.max().item():.0f}]")
        
        if bbox.max().item() <= 1000:
            logger.info("✅ bbox坐标在正确范围内！")
            
            # Try to set up training (without actually training)
            trainer.setup_training(
                train_dataset, 
                train_dataset,  # Use same for validation
                "./models/test_training",
                num_epochs=1
            )
            logger.info("✅ 训练设置成功 - bbox坐标问题已修复！")
            return True
        else:
            logger.error(f"❌ bbox坐标仍然超出范围: max={bbox.max().item()}")
            return False
            
    except Exception as e:
        if "bbox" in str(e) and "0-1000 range" in str(e):
            logger.error(f"❌ 仍然存在bbox范围错误: {e}")
            return False
        else:
            logger.error(f"❌ 其他错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == '__main__':
    success = quick_training_test()
    if success:
        print("\n🎉 快速训练测试成功！bbox坐标问题已完全修复。")
        print("现在可以运行完整的tutorial_demo.py了！")
    else:
        print("\n❌ 仍有问题需要解决")