import time
import psutil
import logging
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, log_file='performance.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def monitor_prediction(self, func):
        """监控预测性能的装饰器"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 记录性能指标
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_usage': end_memory - start_memory,
                'success': success,
                'error': error
            }
            
            self.logger.info(f"Performance metrics: {metrics}")
            
            if success:
                return result
            else:
                raise Exception(error)
        
        return wrapper