import os
import json
from sklearn.model_selection import train_test_split

def main():
    # 1. 数据预处理
    print("开始数据预处理...")
    preprocessor = InvoiceDataPreprocessor("/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data")
    
    # 转换PDF为图像
    pdf_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/測試股份有限公司.pdf"
    image_paths = preprocessor.pdf_to_images(pdf_path)
    
    # 加载标注数据
    with open("/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/label1.json", 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    # 创建训练标注
    annotations = preprocessor.create_training_annotations(label_data, image_paths[0])
    
    # 2. 数据集划分
    train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)
    
    # 保存训练和验证数据
    with open("train_annotations.json", 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, ensure_ascii=False, indent=2)
    
    with open("val_annotations.json", 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, ensure_ascii=False, indent=2)
    
    # 3. 创建数据集
    print("创建数据集...")
    tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
    
    train_dataset = InvoiceLayoutDataset("train_annotations.json", tokenizer)
    val_dataset = InvoiceLayoutDataset("val_annotations.json", tokenizer)
    
    # 4. 训练模型
    print("开始训练模型...")
    trainer = InvoiceLayoutLMTrainer()
    trainer.setup_training(train_dataset, val_dataset, "./invoice_layoutlm_model")
    trainer.train()
    
    # 5. 保存模型
    trainer.save_model("./final_invoice_model")
    
    # 6. 评估模型
    print("评估模型性能...")
    inference_system = InvoiceInferenceSystem("./final_invoice_model")
    evaluator = ModelEvaluator(inference_system)
    
    field_accuracies, predictions, ground_truths = evaluator.evaluate_on_test_set(val_annotations)
    report = evaluator.generate_evaluation_report(field_accuracies, predictions, ground_truths)
    
    # 保存评估报告
    with open("evaluation_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"训练完成！整体准确率: {report['overall_accuracy']:.4f}")
    print("各字段准确率:")
    for field, acc in field_accuracies.items():
        print(f"  {field}: {acc:.4f}")

if __name__ == "__main__":
    main()