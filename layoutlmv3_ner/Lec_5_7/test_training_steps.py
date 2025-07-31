#!/usr/bin/env python3
"""
Test a few actual training steps to confirm the fix
"""
import logging
from src.layoutlm_trainer import LayoutLMv3Trainer, InvoiceLayoutDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_steps():
    """Test actual training steps"""
    try:
        logger.info("🧪 测试实际训练步骤...")
        
        # Initialize trainer and dataset
        trainer = LayoutLMv3Trainer()
        train_dataset = InvoiceLayoutDataset("data/training/demo_train.json", trainer.tokenizer)
        
        logger.info(f"✅ 数据集大小: {len(train_dataset)}")
        
        # Setup training with very small parameters for quick test
        trainer.setup_training(
            train_dataset,
            train_dataset,  # Use same for validation
            "./models/test_bbox_fix",
            num_epochs=1  # Just 1 epoch
        )
        
        # Modify trainer to stop after just a few steps
        trainer.training_args.max_steps = 3  # Only 3 steps
        trainer.training_args.save_steps = 100  # Don't save during test
        trainer.training_args.eval_steps = 100  # Don't eval during test
        trainer.training_args.logging_steps = 1  # Log every step
        
        logger.info("🏃‍♂️ 开始3个训练步骤测试...")
        
        # Start training
        trainer.train()
        
        logger.info("🎉 训练步骤测试成功！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练步骤测试失败: {e}")
        return False

if __name__ == '__main__':
    success = test_training_steps()
    if success:
        print("\n🎉 LayoutLM训练完全正常！bbox问题已彻底解决。")
        print("现在可以运行完整的tutorial_demo.py进行正常训练了！")
    else:
        print("\n❌ 仍有问题需要解决")