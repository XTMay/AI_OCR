#!/usr/bin/env python3
"""
Test just the inference part with JSON saving
"""
import os
import json
import logging
from datetime import datetime
from src.inference_pipeline import InvoiceInferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_json_saving():
    """Test JSON saving functionality"""
    
    model_path = "./models/final_tutorial_invoice_layoutlmv3"
    pdf_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/raw/測試股份有限公司_1.pdf"
    label_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/training/label_1.json"
    
    try:
        logger.info("🧪 测试JSON保存功能...")
        
        # 初始化推理管道
        inference_pipeline = InvoiceInferencePipeline(model_path)
        logger.info("✅ 推理管道初始化成功")
        
        # 创建推理结果保存目录
        inference_results_dir = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/inference_results"
        os.makedirs(inference_results_dir, exist_ok=True)
        logger.info(f"✅ 创建结果目录: {inference_results_dir}")
        
        # 推理
        invoice_info = inference_pipeline.process_invoice(pdf_path)
        logger.info(f"✅ 推理完成: {invoice_info}")
        
        # 保存推理结果到JSON文件
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        inference_result_file = os.path.join(inference_results_dir, f"inference_{pdf_name}.json")
        
        with open(inference_result_file, 'w', encoding='utf-8') as f:
            json.dump(invoice_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 推理结果已保存到: {inference_result_file}")
        
        # 加载ground truth并创建对比
        with open(label_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        comparison_result = {
            "pdf_file": pdf_name,
            "inference_results": invoice_info,
            "ground_truth": ground_truth,
            "field_comparison": {},
            "accuracy_metrics": {}
        }
        
        # 计算准确性
        correct_fields = 0
        total_fields = len(ground_truth)
        
        for field, true_value in ground_truth.items():
            pred_value = invoice_info.get(field, "")
            is_correct = str(pred_value).strip() == str(true_value).strip()
            
            comparison_result["field_comparison"][field] = {
                "predicted": str(pred_value).strip(),
                "ground_truth": str(true_value).strip(),
                "correct": is_correct
            }
            
            if is_correct:
                correct_fields += 1
        
        accuracy = correct_fields / total_fields if total_fields > 0 else 0
        comparison_result["accuracy_metrics"] = {
            "correct_fields": correct_fields,
            "total_fields": total_fields,
            "accuracy": accuracy
        }
        
        # 保存对比结果
        comparison_file = os.path.join(inference_results_dir, f"comparison_{pdf_name}.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 对比结果已保存到: {comparison_file}")
        logger.info(f"🎯 准确率: {accuracy:.2%} ({correct_fields}/{total_fields})")
        
        # 验证文件是否正确保存
        if os.path.exists(inference_result_file) and os.path.exists(comparison_file):
            logger.info("🎉 JSON保存功能测试成功！")
            return True
        else:
            logger.error("❌ JSON文件保存失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_json_saving()
    if success:
        print("\n🎉 JSON保存功能正常工作！")
    else:
        print("\n❌ JSON保存功能有问题")