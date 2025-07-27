#!/usr/bin/env python3
"""
Test if the saved model can be loaded properly for inference
"""
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the saved model"""
    model_path = "./models/final_tutorial_invoice_layoutlmv3"
    
    try:
        logger.info(f"🧪 测试模型加载: {model_path}")
        
        # Check if all required files exist
        required_files = [
            "config.json",
            "model.safetensors", 
            "tokenizer_config.json",
            "preprocessor_config.json"
        ]
        
        logger.info("📋 检查必需文件:")
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                logger.info(f"  ✅ {file}")
            else:
                logger.error(f"  ❌ {file} - 缺失")
                return False
        
        # Try to load the model components
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Tokenizer,
            LayoutLMv3Processor
        )
        
        logger.info("🔧 加载模型组件:")
        
        # Load tokenizer
        tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_path)
        logger.info("  ✅ Tokenizer加载成功")
        
        # Load model
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        logger.info("  ✅ Model加载成功")
        
        # Try to load processor
        try:
            processor = LayoutLMv3Processor.from_pretrained(model_path)
            logger.info("  ✅ Processor加载成功")
        except Exception as e:
            logger.warning(f"  ⚠️ Processor加载失败: {e}")
            logger.info("  🔧 这可能不会影响推理，可以直接使用tokenizer")
        
        logger.info("🎉 模型加载测试成功！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return False

def test_inference_pipeline():
    """Test if inference pipeline can use the model"""
    try:
        logger.info("🔮 测试推理管道...")
        
        from src.inference_pipeline import InvoiceInferencePipeline
        
        # Initialize inference pipeline
        model_path = "./models/final_tutorial_invoice_layoutlmv3"
        
        pipeline = InvoiceInferencePipeline(model_path)
        logger.info("  ✅ 推理管道初始化成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 推理管道测试失败: {e}")
        return False

if __name__ == '__main__':
    logger.info("🧪 开始模型加载和推理测试...")
    
    # Test model loading
    loading_success = test_model_loading()
    
    # Test inference pipeline if loading succeeded
    if loading_success:
        inference_success = test_inference_pipeline()
        
        if inference_success:
            print("\n🎉 所有测试通过！模型可以正常用于推理。")
            print("现在可以继续运行tutorial_demo.py的推理部分了。")
        else:
            print("\n⚠️ 模型加载成功，但推理管道有问题")
    else:
        print("\n❌ 模型加载失败，需要进一步修复")