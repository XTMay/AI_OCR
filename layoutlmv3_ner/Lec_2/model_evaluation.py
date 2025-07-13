import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, inference_system):
        self.inference_system = inference_system
        
    def evaluate_on_test_set(self, test_annotations):
        """在测试集上评估模型"""
        predictions = []
        ground_truths = []
        
        for annotation in test_annotations:
            # 预测
            pred_result = self.inference_system.predict_invoice(annotation['image'])
            predictions.append(pred_result)
            
            # 真实标签
            ground_truths.append(annotation['ground_truth'])
        
        # 计算字段级别的准确率
        field_accuracies = {}
        fields = ['InvoiceNo', 'InvoiceDate', 'Currency', 'Amount with Tax', 'Amount without Tax', 'Tax']
        
        for field in fields:
            correct = 0
            total = 0
            
            for pred, gt in zip(predictions, ground_truths):
                if field in gt and gt[field]:  # 只评估有标注的字段
                    total += 1
                    if pred.get(field, '') == gt[field]:
                        correct += 1
            
            if total > 0:
                field_accuracies[field] = correct / total
            else:
                field_accuracies[field] = 0.0
        
        return field_accuracies, predictions, ground_truths
    
    def generate_evaluation_report(self, field_accuracies, predictions, ground_truths):
        """生成评估报告"""
        report = {
            'overall_accuracy': sum(field_accuracies.values()) / len(field_accuracies),
            'field_accuracies': field_accuracies,
            'total_samples': len(predictions),
            'detailed_results': []
        }
        
        # 详细结果
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            sample_result = {
                'sample_id': i,
                'prediction': pred,
                'ground_truth': gt,
                'field_matches': {}
            }
            
            for field in ['InvoiceNo', 'InvoiceDate', 'Currency', 'Amount with Tax', 'Amount without Tax', 'Tax']:
                pred_val = pred.get(field, '')
                gt_val = gt.get(field, '')
                sample_result['field_matches'][field] = pred_val == gt_val
            
            report['detailed_results'].append(sample_result)
        
        return report