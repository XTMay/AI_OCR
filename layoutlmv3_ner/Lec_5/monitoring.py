import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, asdict
import os
from pathlib import Path

@dataclass
class ExtractionMetrics:
    """提取指标数据类"""
    timestamp: datetime
    processing_time: float
    success: bool
    extracted_fields: Dict[str, Any]
    confidence_scores: Dict[str, float]
    error_message: Optional[str] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None

class InvoiceExtractionMonitor:
    """发票提取性能监控"""
    
    def __init__(self, log_file: str = "extraction_monitor.log", 
                 metrics_file: str = "metrics_history.json"):
        self.metrics = {
            'total_requests': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_processing_time': 0,
            'field_accuracy': {}
        }
        
        # 详细指标历史
        self.metrics_history: deque = deque(maxlen=10000)  # 保留最近10000条记录
        self.field_statistics = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_extractions': 0,
            'confidence_sum': 0,
            'accuracy_rate': 0
        })
        
        # 实时性能监控
        self.recent_requests = deque(maxlen=100)  # 最近100个请求
        self.performance_alerts = []
        
        # 文件路径
        self.log_file = log_file
        self.metrics_file = metrics_file
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 设置日志
        self.setup_logging()
        
        # 加载历史数据
        self.load_metrics_history()
        
        # 性能阈值
        self.performance_thresholds = {
            'max_processing_time': 30.0,  # 秒
            'min_success_rate': 0.95,     # 95%
            'max_memory_usage': 80,       # 80%
            'max_cpu_usage': 90           # 90%
        }
    
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_extraction_result(self, result: Dict, processing_time: float, 
                            file_info: Optional[Dict] = None):
        """记录提取结果"""
        with self.lock:
            self.metrics['total_requests'] += 1
            
            # 创建详细指标记录
            metrics_record = ExtractionMetrics(
                timestamp=datetime.now(),
                processing_time=processing_time,
                success=result.get('success', False),
                extracted_fields=result.get('extracted_data', {}),
                confidence_scores=result.get('confidence_scores', {}),
                error_message=result.get('error_message'),
                file_size=file_info.get('file_size') if file_info else None,
                page_count=file_info.get('page_count') if file_info else None
            )
            
            # 添加到历史记录
            self.metrics_history.append(metrics_record)
            self.recent_requests.append(metrics_record)
            
            if result.get('success'):
                self.metrics['successful_extractions'] += 1
                self._update_field_statistics(result.get('extracted_data', {}), 
                                            result.get('confidence_scores', {}))
            else:
                self.metrics['failed_extractions'] += 1
                self.logger.warning(f"Extraction failed: {result.get('error_message')}")
            
            # 更新平均处理时间
            self.metrics['average_processing_time'] = (
                (self.metrics['average_processing_time'] * (self.metrics['total_requests'] - 1) + processing_time) 
                / self.metrics['total_requests']
            )
            
            # 检查性能警报
            self._check_performance_alerts(metrics_record)
            
            # 定期保存指标
            if self.metrics['total_requests'] % 100 == 0:
                self.save_metrics_history()
    
    def _update_field_statistics(self, extracted_data: Dict, confidence_scores: Dict):
        """更新字段统计信息"""
        for field_name, field_value in extracted_data.items():
            stats = self.field_statistics[field_name]
            stats['total_attempts'] += 1
            
            if field_value and field_value != "":
                stats['successful_extractions'] += 1
                
                # 更新置信度
                confidence = confidence_scores.get(field_name, 0)
                stats['confidence_sum'] += confidence
            
            # 计算准确率
            stats['accuracy_rate'] = stats['successful_extractions'] / stats['total_attempts']
            
            # 更新全局字段准确率
            self.metrics['field_accuracy'][field_name] = stats['accuracy_rate']
    
    def _check_performance_alerts(self, metrics_record: ExtractionMetrics):
        """检查性能警报"""
        alerts = []
        
        # 检查处理时间
        if metrics_record.processing_time > self.performance_thresholds['max_processing_time']:
            alerts.append(f"High processing time: {metrics_record.processing_time:.2f}s")
        
        # 检查成功率（最近100个请求）
        if len(self.recent_requests) >= 10:
            recent_success_rate = sum(1 for r in self.recent_requests if r.success) / len(self.recent_requests)
            if recent_success_rate < self.performance_thresholds['min_success_rate']:
                alerts.append(f"Low success rate: {recent_success_rate:.2%}")
        
        # 检查系统资源
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        if memory_usage > self.performance_thresholds['max_memory_usage']:
            alerts.append(f"High memory usage: {memory_usage:.1f}%")
        
        if cpu_usage > self.performance_thresholds['max_cpu_usage']:
            alerts.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        # 记录警报
        for alert in alerts:
            self.performance_alerts.append({
                'timestamp': metrics_record.timestamp,
                'alert': alert
            })
            self.logger.warning(f"Performance Alert: {alert}")
    
    def get_current_metrics(self) -> Dict:
        """获取当前指标"""
        with self.lock:
            current_metrics = self.metrics.copy()
            
            # 添加成功率
            if current_metrics['total_requests'] > 0:
                current_metrics['success_rate'] = (
                    current_metrics['successful_extractions'] / current_metrics['total_requests']
                )
            else:
                current_metrics['success_rate'] = 0
            
            # 添加系统资源信息
            current_metrics['system_info'] = {
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent
            }
            
            return current_metrics
    
    def get_field_performance(self) -> Dict:
        """获取字段级别性能统计"""
        with self.lock:
            field_performance = {}
            
            for field_name, stats in self.field_statistics.items():
                avg_confidence = (
                    stats['confidence_sum'] / stats['successful_extractions'] 
                    if stats['successful_extractions'] > 0 else 0
                )
                
                field_performance[field_name] = {
                    'accuracy_rate': stats['accuracy_rate'],
                    'total_attempts': stats['total_attempts'],
                    'successful_extractions': stats['successful_extractions'],
                    'average_confidence': avg_confidence
                }
            
            return field_performance
    
    def get_time_series_metrics(self, hours: int = 24) -> List[Dict]:
        """获取时间序列指标"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        time_series = []
        for record in self.metrics_history:
            if record.timestamp >= cutoff_time:
                time_series.append({
                    'timestamp': record.timestamp.isoformat(),
                    'processing_time': record.processing_time,
                    'success': record.success,
                    'confidence_scores': record.confidence_scores
                })
        
        return time_series
    
    def generate_performance_report(self, output_dir: str = "reports") -> str:
        """生成性能报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建报告数据
        report_data = {
            'generation_time': datetime.now().isoformat(),
            'summary_metrics': self.get_current_metrics(),
            'field_performance': self.get_field_performance(),
            'recent_alerts': self.performance_alerts[-50:],  # 最近50个警报
            'time_series_data': self.get_time_series_metrics(24)
        }
        
        # 保存JSON报告
        report_file = os.path.join(output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成可视化图表
        self._generate_performance_charts(output_dir)
        
        self.logger.info(f"Performance report generated: {report_file}")
        return report_file
    
    def _generate_performance_charts(self, output_dir: str):
        """生成性能图表"""
        if not self.metrics_history:
            return
        
        # 准备数据
        df_data = []
        for record in self.metrics_history:
            df_data.append({
                'timestamp': record.timestamp,
                'processing_time': record.processing_time,
                'success': record.success,
                'hour': record.timestamp.hour
            })
        
        df = pd.DataFrame(df_data)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 处理时间趋势
        axes[0, 0].plot(df['timestamp'], df['processing_time'])
        axes[0, 0].set_title('Processing Time Trend')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Processing Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 成功率趋势
        success_rate = df.groupby(df['timestamp'].dt.hour)['success'].mean()
        axes[0, 1].bar(success_rate.index, success_rate.values)
        axes[0, 1].set_title('Success Rate by Hour')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Success Rate')
        
        # 3. 字段准确率
        field_perf = self.get_field_performance()
        if field_perf:
            fields = list(field_perf.keys())
            accuracies = [field_perf[field]['accuracy_rate'] for field in fields]
            
            axes[1, 0].barh(fields, accuracies)
            axes[1, 0].set_title('Field Accuracy Rates')
            axes[1, 0].set_xlabel('Accuracy Rate')
        
        # 4. 处理时间分布
        axes[1, 1].hist(df['processing_time'], bins=30, alpha=0.7)
        axes[1, 1].set_title('Processing Time Distribution')
        axes[1, 1].set_xlabel('Processing Time (s)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance charts saved: {chart_file}")
    
    def save_metrics_history(self):
        """保存指标历史"""
        try:
            # 转换为可序列化格式
            serializable_history = []
            for record in self.metrics_history:
                record_dict = asdict(record)
                record_dict['timestamp'] = record.timestamp.isoformat()
                serializable_history.append(record_dict)
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': self.metrics,
                    'field_statistics': dict(self.field_statistics),
                    'history': serializable_history
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Metrics history saved to {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics history: {e}")
    
    def load_metrics_history(self):
        """加载指标历史"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 恢复基本指标
                self.metrics.update(data.get('metrics', {}))
                
                # 恢复字段统计
                field_stats = data.get('field_statistics', {})
                for field, stats in field_stats.items():
                    self.field_statistics[field].update(stats)
                
                # 恢复历史记录
                history = data.get('history', [])
                for record_dict in history:
                    record_dict['timestamp'] = datetime.fromisoformat(record_dict['timestamp'])
                    record = ExtractionMetrics(**record_dict)
                    self.metrics_history.append(record)
                
                self.logger.info(f"Loaded {len(history)} historical records")
        except Exception as e:
            self.logger.warning(f"Failed to load metrics history: {e}")
    
    def reset_metrics(self):
        """重置所有指标"""
        with self.lock:
            self.metrics = {
                'total_requests': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'average_processing_time': 0,
                'field_accuracy': {}
            }
            self.metrics_history.clear()
            self.field_statistics.clear()
            self.recent_requests.clear()
            self.performance_alerts.clear()
            
            self.logger.info("All metrics have been reset")
    
    def get_performance_summary(self) -> str:
        """获取性能摘要文本"""
        metrics = self.get_current_metrics()
        field_perf = self.get_field_performance()
        
        summary = f"""
=== Invoice Extraction Performance Summary ===

Overall Metrics:
- Total Requests: {metrics['total_requests']}
- Success Rate: {metrics.get('success_rate', 0):.2%}
- Average Processing Time: {metrics['average_processing_time']:.2f}s
- Failed Extractions: {metrics['failed_extractions']}

System Resources:
- Memory Usage: {metrics['system_info']['memory_usage']:.1f}%
- CPU Usage: {metrics['system_info']['cpu_usage']:.1f}%
- Disk Usage: {metrics['system_info']['disk_usage']:.1f}%

Field Performance:
"""
        
        for field, perf in field_perf.items():
            summary += f"- {field}: {perf['accuracy_rate']:.2%} accuracy ({perf['total_attempts']} attempts)\n"
        
        if self.performance_alerts:
            summary += f"\nRecent Alerts: {len(self.performance_alerts)} alerts in history\n"
        
        return summary