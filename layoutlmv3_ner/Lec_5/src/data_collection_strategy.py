class InvoiceDataCollectionStrategy:
    """发票数据收集策略"""
    
    def __init__(self):
        self.invoice_types = {
            "增值税专用发票": {"priority": "high", "samples_needed": 200},
            "增值税普通发票": {"priority": "high", "samples_needed": 150},
            "电子发票": {"priority": "medium", "samples_needed": 100},
            "国际发票": {"priority": "medium", "samples_needed": 80},
            "手写发票": {"priority": "low", "samples_needed": 50}
        }
        
        self.language_distribution = {
            "中文": 0.6,
            "英文": 0.3,
            "中英混合": 0.1
        }
    
    def generate_collection_plan(self):
        """生成数据收集计划"""
        plan = {
            "total_samples": sum(info["samples_needed"] for info in self.invoice_types.values()),
            "collection_phases": [
                {"phase": 1, "focus": "高频发票类型", "duration": "2周"},
                {"phase": 2, "focus": "多语言样本", "duration": "1周"},
                {"phase": 3, "focus": "边缘案例", "duration": "1周"}
            ],
            "quality_criteria": {
                "resolution": ">=300 DPI",
                "clarity": "清晰可读",
                "completeness": "包含所有必要字段"
            }
        }
        return plan