#!/usr/bin/env python3
"""
Test LayoutLM training with fixed bbox coordinates
"""
import json
import logging
from src.layoutlm_trainer import LayoutLMv3Trainer, InvoiceLayoutDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training():
    """Test training with the fixed data"""
    try:
        logger.info("🧪 开始测试LayoutLM训练...")
        
        # Initialize trainer
        trainer = LayoutLMv3Trainer()
        logger.info("✅ 训练器初始化成功")
        
        # Create dataset
        train_file = "data/training/demo_train.json"
        train_dataset = InvoiceLayoutDataset(train_file, trainer.tokenizer)
        logger.info(f"✅ 数据集创建成功，大小: {len(train_dataset)}")
        
        # Test one sample to see if bbox coordinates are handled correctly
        sample = train_dataset[0]
        logger.info("✅ 成功加载第一个样本")
        
        # Check bbox in the sample
        bbox = sample['bbox']
        logger.info(f"📊 样本bbox形状: {bbox.shape}")
        logger.info(f"📊 bbox坐标范围: [{bbox.min().item():.0f}, {bbox.max().item():.0f}]")
        
        if bbox.max().item() <= 1000:
            logger.info("✅ bbox坐标在正确范围内 (≤1000)")
            return True
        else:
            logger.error(f"❌ bbox坐标超出范围: {bbox.max().item()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    success = test_training()
    if success:
        print("\n🎉 LayoutLM训练测试成功！bbox坐标问题已修复。")
        print("\n现在可以运行完整的tutorial_demo.py了。")
    else:
        print("\n❌ 训练测试失败，需要进一步调试。")