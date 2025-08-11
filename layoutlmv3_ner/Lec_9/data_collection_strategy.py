class DataCollectionStrategy:
    def __init__(self):
        self.data_sources = {
            'real_invoices': [],
            'synthetic_data': [],
            'augmented_data': [],
            'cross_domain_data': []
        }
    
    def collect_diverse_samples(self):
        """收集多样化样本"""
        collection_plan = {
            # 不同行业发票
            'industries': [
                'retail', 'manufacturing', 'service', 
                'healthcare', 'education', 'finance'
            ],
            
            # 不同格式
            'formats': [
                'standard_invoice', 'receipt', 'tax_invoice',
                'credit_note', 'debit_note', 'proforma'
            ],
            
            # 不同质量等级
            'quality_levels': [
                'high_quality', 'medium_quality', 'low_quality',
                'scanned', 'photographed', 'faxed'
            ],
            
            # 不同语言
            'languages': [
                'chinese', 'english', 'japanese', 'korean',
                'spanish', 'french', 'german', 'arabic'
            ]
        }
        
        return collection_plan
    
    def balance_dataset(self, dataset):
        """数据集平衡"""
        # 分析各字段的分布
        field_distribution = self.analyze_field_distribution(dataset)
        
        # 生成平衡策略
        balance_strategy = self.create_balance_strategy(field_distribution)
        
        # 应用平衡策略
        balanced_dataset = self.apply_balance_strategy(dataset, balance_strategy)
        
        return balanced_dataset