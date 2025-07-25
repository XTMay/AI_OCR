import re
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

class InvoiceModelEvaluator:
    """发票模型评估器"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        初始化评估器
        
        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_extraction_performance(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """评估信息提取性能"""
        metrics = {
            'field_accuracy': {},
            'overall_accuracy': 0.0,
            'precision': {},
            'recall': {},
            'f1_score': {},
            'confusion_matrix': {},
            'error_analysis': {}
        }
        
        fields = ['InvoiceNo', 'InvoiceDate', 'Currency', 'Amount with Tax', 'Amount without Tax', 'Tax']
        
        for field in fields:
            tp = fp = fn = tn = 0
            errors = []
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                pred_value = pred.get(field, '').strip()
                gt_value = gt.get(field, '').strip()
                
                if pred_value and gt_value:
                    if self._is_match(pred_value, gt_value, field):
                        tp += 1
                    else:
                        fp += 1
                        errors.append({
                            'index': i,
                            'predicted': pred_value,
                            'ground_truth': gt_value,
                            'error_type': 'wrong_value'
                        })
                elif pred_value and not gt_value:
                    fp += 1
                    errors.append({
                        'index': i,
                        'predicted': pred_value,
                        'ground_truth': gt_value,
                        'error_type': 'false_positive'
                    })
                elif not pred_value and gt_value:
                    fn += 1
                    errors.append({
                        'index': i,
                        'predicted': pred_value,
                        'ground_truth': gt_value,
                        'error_type': 'false_negative'
                    })
                else:
                    tn += 1
            
            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            metrics['precision'][field] = precision
            metrics['recall'][field] = recall
            metrics['f1_score'][field] = f1
            metrics['field_accuracy'][field] = accuracy
            metrics['confusion_matrix'][field] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            }
            metrics['error_analysis'][field] = errors
        
        # 计算整体指标
        metrics['overall_accuracy'] = sum(metrics['f1_score'].values()) / len(fields)
        metrics['macro_precision'] = sum(metrics['precision'].values()) / len(fields)
        metrics['macro_recall'] = sum(metrics['recall'].values()) / len(fields)
        metrics['macro_f1'] = sum(metrics['f1_score'].values()) / len(fields)
        
        return metrics
    
    def _is_match(self, pred: str, gt: str, field_type: str) -> bool:
        """字段特定的匹配逻辑"""
        if field_type == 'InvoiceDate':
            # 日期匹配：提取数字部分比较
            pred_nums = re.findall(r'\d+', pred)
            gt_nums = re.findall(r'\d+', gt)
            return pred_nums == gt_nums
        
        elif field_type in ['Amount with Tax', 'Amount without Tax', 'Tax']:
            # 金额匹配：提取数字部分比较
            pred_amount = re.sub(r'[^\d.]', '', pred)
            gt_amount = re.sub(r'[^\d.]', '', gt)
            try:
                return abs(float(pred_amount) - float(gt_amount)) < 0.01
            except ValueError:
                return pred.lower() == gt.lower()
        
        else:
            # 文本匹配：忽略大小写和空格
            return pred.lower().replace(' ', '') == gt.lower().replace(' ', '')
    
    def evaluate_token_classification(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """评估token级别的分类性能"""
        # 确保预测和真实标签长度一致
        min_len = min(len(predictions), len(ground_truths))
        predictions = predictions[:min_len]
        ground_truths = ground_truths[:min_len]
        
        # 生成分类报告
        report = classification_report(
            ground_truths, 
            predictions, 
            output_dict=True,
            zero_division=0
        )
        
        # 生成混淆矩阵
        labels = sorted(list(set(ground_truths + predictions)))
        cm = confusion_matrix(ground_truths, predictions, labels=labels)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'labels': labels,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def analyze_errors(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """详细错误分析"""
        error_analysis = {
            'field_errors': defaultdict(list),
            'common_mistakes': defaultdict(int),
            'error_patterns': defaultdict(list),
            'confidence_analysis': {}
        }
        
        fields = ['InvoiceNo', 'InvoiceDate', 'Currency', 'Amount with Tax', 'Amount without Tax', 'Tax']
        
        for field in fields:
            field_errors = []
            confidences = []
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                pred_value = pred.get(field, '').strip()
                gt_value = gt.get(field, '').strip()
                confidence = pred.get(f'{field}_confidence', 1.0)
                confidences.append(confidence)
                
                if not self._is_match(pred_value, gt_value, field):
                    error_type = self._classify_error(pred_value, gt_value, field)
                    field_errors.append({
                        'index': i,
                        'predicted': pred_value,
                        'ground_truth': gt_value,
                        'error_type': error_type,
                        'confidence': confidence
                    })
                    
                    # 统计常见错误
                    error_analysis['common_mistakes'][error_type] += 1
            
            error_analysis['field_errors'][field] = field_errors
            error_analysis['confidence_analysis'][field] = {
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'low_confidence_threshold': np.percentile(confidences, 25)
            }
        
        return dict(error_analysis)
    
    def _classify_error(self, pred: str, gt: str, field_type: str) -> str:
        """分类错误类型"""
        if not pred and gt:
            return 'missing_extraction'
        elif pred and not gt:
            return 'false_extraction'
        elif pred and gt:
            if field_type == 'InvoiceDate':
                if any(char.isdigit() for char in pred) and any(char.isdigit() for char in gt):
                    return 'date_format_error'
                else:
                    return 'date_content_error'
            elif field_type in ['Amount with Tax', 'Amount without Tax', 'Tax']:
                try:
                    pred_num = float(re.sub(r'[^\d.]', '', pred))
                    gt_num = float(re.sub(r'[^\d.]', '', gt))
                    if abs(pred_num - gt_num) / gt_num > 0.1:
                        return 'amount_significant_error'
                    else:
                        return 'amount_minor_error'
                except:
                    return 'amount_format_error'
            else:
                return 'text_content_error'
        else:
            return 'unknown_error'
    
    def generate_performance_report(self, metrics: Dict, save_path: str = None) -> str:
        """生成性能报告"""
        report = []
        report.append("# 发票信息提取性能评估报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## 整体性能指标")
        report.append(f"- 整体准确率: {metrics['overall_accuracy']:.4f}")
        report.append(f"- 宏平均精确率: {metrics.get('macro_precision', 0):.4f}")
        report.append(f"- 宏平均召回率: {metrics.get('macro_recall', 0):.4f}")
        report.append(f"- 宏平均F1分数: {metrics.get('macro_f1', 0):.4f}")
        
        report.append("\n## 各字段性能详情")
        for field in ['InvoiceNo', 'InvoiceDate', 'Currency', 'Amount with Tax', 'Amount without Tax', 'Tax']:
            if field in metrics['precision']:
                report.append(f"\n### {field}")
                report.append(f"- 精确率: {metrics['precision'][field]:.4f}")
                report.append(f"- 召回率: {metrics['recall'][field]:.4f}")
                report.append(f"- F1分数: {metrics['f1_score'][field]:.4f}")
                report.append(f"- 准确率: {metrics['field_accuracy'][field]:.4f}")
                
                # 混淆矩阵
                cm = metrics['confusion_matrix'][field]
                report.append(f"- 真正例: {cm['tp']}, 假正例: {cm['fp']}")
                report.append(f"- 假负例: {cm['fn']}, 真负例: {cm['tn']}")
        
        report.append("\n## 错误分析")
        if 'error_analysis' in metrics:
            for field, errors in metrics['error_analysis'].items():
                if errors:
                    report.append(f"\n### {field} 错误统计")
                    error_types = Counter([error['error_type'] for error in errors])
                    for error_type, count in error_types.items():
                        report.append(f"- {error_type}: {count}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Performance report saved to {save_path}")
        
        return report_text
    
    def plot_performance_metrics(self, metrics: Dict, save_path: str = None):
        """绘制性能指标图表"""
        fields = ['InvoiceNo', 'InvoiceDate', 'Currency', 'Amount with Tax', 'Amount without Tax', 'Tax']
        
        # 准备数据
        precision_scores = [metrics['precision'].get(field, 0) for field in fields]
        recall_scores = [metrics['recall'].get(field, 0) for field in fields]
        f1_scores = [metrics['f1_score'].get(field, 0) for field in fields]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 各字段性能对比
        x = np.arange(len(fields))
        width = 0.25
        
        axes[0, 0].bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recall_scores, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[0, 0].set_xlabel('Fields')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance by Field')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(fields, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 混淆矩阵热图（以第一个字段为例）
        if fields and fields[0] in metrics['confusion_matrix']:
            cm_data = metrics['confusion_matrix'][fields[0]]
            cm_matrix = np.array([[cm_data['tp'], cm_data['fp']], 
                                 [cm_data['fn'], cm_data['tn']]])
            sns.heatmap(cm_matrix, annot=True, fmt='d', 
                       xticklabels=['Predicted Positive', 'Predicted Negative'],
                       yticklabels=['Actual Positive', 'Actual Negative'],
                       ax=axes[0, 1])
            axes[0, 1].set_title(f'Confusion Matrix - {fields[0]}')
        
        # 3. F1分数分布
        axes[1, 0].bar(fields, f1_scores, color='skyblue', alpha=0.7)
        axes[1, 0].set_xlabel('Fields')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 错误类型分布
        if 'error_analysis' in metrics:
            error_counts = defaultdict(int)
            for field, errors in metrics['error_analysis'].items():
                for error in errors:
                    error_counts[error['error_type']] += 1
            
            if error_counts:
                error_types = list(error_counts.keys())
                error_values = list(error_counts.values())
                axes[1, 1].pie(error_values, labels=error_types, autopct='%1.1f%%')
                axes[1, 1].set_title('Error Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance plots saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """比较多个模型的性能"""
        comparison = {
            'model_names': list(model_results.keys()),
            'overall_scores': {},
            'field_scores': defaultdict(dict),
            'best_model': {}
        }
        
        fields = ['InvoiceNo', 'InvoiceDate', 'Currency', 'Amount with Tax', 'Amount without Tax', 'Tax']
        
        # 收集各模型的整体分数
        for model_name, metrics in model_results.items():
            comparison['overall_scores'][model_name] = metrics.get('overall_accuracy', 0)
            
            # 收集各字段分数
            for field in fields:
                comparison['field_scores'][field][model_name] = metrics.get('f1_score', {}).get(field, 0)
        
        # 找出最佳模型
        best_overall = max(comparison['overall_scores'], key=comparison['overall_scores'].get)
        comparison['best_model']['overall'] = best_overall
        
        for field in fields:
            if field in comparison['field_scores']:
                best_field = max(comparison['field_scores'][field], 
                               key=comparison['field_scores'][field].get)
                comparison['best_model'][field] = best_field
        
        return dict(comparison)
    
    def save_evaluation_results(self, metrics: Dict, filename: str = None):
        """保存评估结果"""
        if filename is None:
            filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = f"{self.output_dir}/{filename}"
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_metrics = convert_numpy(metrics)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_metrics, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Evaluation results saved to {filepath}")
        return filepath