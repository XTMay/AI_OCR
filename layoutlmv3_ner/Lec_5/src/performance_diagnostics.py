import numpy as np
import os  # Add this import
from typing import Dict, List, Tuple, Any
import logging

class PerformanceDiagnostics:
    """性能问题诊断工具"""
    
    def __init__(self):
        self.diagnostic_rules = {
            "low_f1_score": {
                "threshold": 0.7,
                "description": "F1分数过低",
                "possible_causes": [
                    "训练数据不足",
                    "数据质量问题",
                    "模型复杂度不够",
                    "学习率设置不当",
                    "类别不平衡"
                ],
                "solutions": [
                    "增加训练数据",
                    "改进数据标注质量",
                    "使用更大的预训练模型",
                    "调整学习率",
                    "使用类别权重或focal loss"
                ]
            },
            "overfitting": {
                "description": "过拟合",
                "possible_causes": [
                    "训练数据过少",
                    "模型过于复杂",
                    "正则化不足",
                    "训练轮数过多"
                ],
                "solutions": [
                    "增加训练数据",
                    "使用dropout和权重衰减",
                    "早停策略",
                    "数据增强",
                    "减少模型复杂度"
                ]
            },
            "slow_convergence": {
                "description": "收敛缓慢",
                "possible_causes": [
                    "学习率过小",
                    "梯度消失",
                    "数据预处理问题",
                    "批次大小不当"
                ],
                "solutions": [
                    "增加学习率",
                    "使用梯度裁剪",
                    "改进数据归一化",
                    "调整批次大小",
                    "使用学习率预热"
                ]
            },
            "unstable_training": {
                "description": "训练不稳定",
                "possible_causes": [
                    "学习率过大",
                    "梯度爆炸",
                    "数据噪声过多",
                    "批次大小过小"
                ],
                "solutions": [
                    "降低学习率",
                    "梯度裁剪",
                    "数据清洗",
                    "增加批次大小",
                    "使用更稳定的优化器"
                ]
            }
        }
    
    def diagnose_training_issues(self, training_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """诊断训练问题"""
        issues = []
        recommendations = []
        
        # 检查F1分数
        if training_history.get('eval_f1'):
            max_f1 = max(training_history['eval_f1'])
            if max_f1 < self.diagnostic_rules['low_f1_score']['threshold']:
                issues.append(self.diagnostic_rules['low_f1_score'])
        
        # 检查过拟合
        overfitting_detected = self._detect_overfitting(training_history)
        if overfitting_detected:
            issues.append(self.diagnostic_rules['overfitting'])
        
        # 检查收敛速度
        slow_convergence = self._detect_slow_convergence(training_history)
        if slow_convergence:
            issues.append(self.diagnostic_rules['slow_convergence'])
        
        # 检查训练稳定性
        unstable_training = self._detect_unstable_training(training_history)
        if unstable_training:
            issues.append(self.diagnostic_rules['unstable_training'])
        
        # 生成建议
        for issue in issues:
            recommendations.extend(issue['solutions'])
        
        return {
            "detected_issues": [issue['description'] for issue in issues],
            "detailed_analysis": issues,
            "recommendations": list(set(recommendations)),  # 去重
            "severity": self._assess_severity(issues)
        }
    
    def _detect_overfitting(self, history: Dict[str, List[float]], lookback: int = 5) -> bool:
        """检测过拟合"""
        if not all(key in history for key in ['train_loss', 'eval_loss']):
            return False
        
        if len(history['train_loss']) < lookback * 2:
            return False
        
        # 计算最近几轮的趋势
        recent_train = history['train_loss'][-lookback:]
        recent_eval = history['eval_loss'][-lookback:]
        
        # 训练损失下降，验证损失上升
        train_decreasing = recent_train[-1] < recent_train[0]
        eval_increasing = recent_eval[-1] > recent_eval[0]
        
        # 验证损失明显高于训练损失
        loss_gap = np.mean(recent_eval) - np.mean(recent_train)
        
        return train_decreasing and eval_increasing and loss_gap > 0.1
    
    def _detect_slow_convergence(self, history: Dict[str, List[float]], patience: int = 10) -> bool:
        """检测收敛缓慢"""
        if 'eval_f1' not in history or len(history['eval_f1']) < patience:
            return False
        
        recent_f1 = history['eval_f1'][-patience:]
        improvement = max(recent_f1) - min(recent_f1)
        
        return improvement < 0.01  # F1分数改善小于1%
    
    def _detect_unstable_training(self, history: Dict[str, List[float]]) -> bool:
        """检测训练不稳定"""
        if 'train_loss' not in history or len(history['train_loss']) < 10:
            return False
        
        # 计算损失的变异系数
        train_loss = np.array(history['train_loss'])
        cv = np.std(train_loss) / np.mean(train_loss)
        
        return cv > 0.3  # 变异系数大于30%
    
    def _assess_severity(self, issues: List[Dict]) -> str:
        """评估问题严重程度"""
        if len(issues) == 0:
            return "无问题"
        elif len(issues) == 1:
            return "轻微"
        elif len(issues) == 2:
            return "中等"
        else:
            return "严重"
    
    def generate_optimization_suggestions(self, current_config: Dict, issues: List[str]) -> Dict[str, Any]:
        """生成优化建议"""
        suggestions = {
            "hyperparameter_adjustments": {},
            "architecture_changes": [],
            "data_improvements": [],
            "training_strategy": []
        }
        
        if "F1分数过低" in issues:
            suggestions["hyperparameter_adjustments"].update({
                "learning_rate": "尝试 1e-5 到 5e-5 之间的值",
                "batch_size": "增加到 4 或 8",
                "epochs": "增加到 20-30 轮"
            })
            suggestions["data_improvements"].extend([
                "增加训练样本数量",
                "改进数据标注质量",
                "平衡各类别样本数量"
            ])
        
        if "过拟合" in issues:
            suggestions["hyperparameter_adjustments"].update({
                "weight_decay": "增加到 0.01-0.1",
                "dropout_rate": "设置为 0.1-0.3"
            })
            suggestions["training_strategy"].extend([
                "使用早停策略",
                "增加数据增强",
                "使用交叉验证"
            ])
        
        if "收敛缓慢" in issues:
            suggestions["hyperparameter_adjustments"].update({
                "learning_rate": "适当增加学习率",
                "warmup_steps": "增加预热步数到 1000"
            })
            suggestions["training_strategy"].extend([
                "使用学习率调度器",
                "梯度累积"
            ])
        
        if "训练不稳定" in issues:
            suggestions["hyperparameter_adjustments"].update({
                "learning_rate": "降低学习率",
                "gradient_clip_norm": "设置梯度裁剪为 1.0"
            })
            suggestions["training_strategy"].extend([
                "使用更稳定的优化器 (AdamW)",
                "增加批次大小"
            ])
        
        return suggestions
    
    def analyze_ocr_performance(self, pdf_path: str) -> Dict[str, Any]:
        """分析OCR性能"""
        try:
            # 基本OCR性能指标
            metrics = {
                "文件路径": pdf_path,
                "文件大小": f"{os.path.getsize(pdf_path) / 1024:.2f} KB" if os.path.exists(pdf_path) else "文件不存在",
                "OCR引擎": "EasyOCR",
                "支持语言": "中文简体, 英文",
                "预期准确率": "85-95%",
                "处理状态": "就绪"
            }
            
            # 如果文件存在，添加更多信息
            if os.path.exists(pdf_path):
                metrics.update({
                    "文件状态": "存在",
                    "建议优化": [
                        "确保图像清晰度足够",
                        "检查文字方向是否正确",
                        "验证语言设置匹配文档内容"
                    ]
                })
            else:
                metrics.update({
                    "文件状态": "不存在",
                    "错误信息": "无法找到指定的PDF文件"
                })
            
            return metrics
            
        except Exception as e:
            return {
                "错误": f"OCR性能分析失败: {str(e)}",
                "文件路径": pdf_path,
                "状态": "分析失败"
            }