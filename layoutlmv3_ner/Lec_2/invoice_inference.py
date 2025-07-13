import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import json
import re
from datetime import datetime

class InvoiceInferenceSystem:
    def __init__(self, model_path):
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
        self.model = LayoutLMForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        # 标签映射（反向）
        self.id_to_label = {
            0: 'O',
            1: 'B-InvoiceNo',
            2: 'I-InvoiceNo',
            3: 'B-InvoiceDate',
            4: 'I-InvoiceDate',
            5: 'B-Currency',
            6: 'I-Currency',
            7: 'B-AmountWithTax',
            8: 'I-AmountWithTax',
            9: 'B-AmountWithoutTax',
            10: 'I-AmountWithoutTax',
            11: 'B-Tax',
            12: 'I-Tax'
        }
        
        # 初始化OCR
        self.ocr_processor = AdvancedPaddleOCR()
        
    def predict_invoice(self, image_path):
        """预测发票信息"""
        # 1. OCR提取文本和布局
        text_blocks = self.ocr_processor.extract_text_with_layout(image_path)
        
        # 2. 准备LayoutLM输入
        words = []
        boxes = []
        
        for block in text_blocks:
            word_tokens = self.tokenizer.tokenize(block['text'])
            words.extend(word_tokens)
            
            # 为每个token分配边界框
            for _ in word_tokens:
                boxes.append([
                    int(block['bbox']['x_min']),
                    int(block['bbox']['y_min']),
                    int(block['bbox']['x_max']),
                    int(block['bbox']['y_max'])
                ])
        
        # 3. 编码输入
        encoding = self.tokenizer(
            words,
            boxes=boxes,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 4. 模型预测
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # 5. 解析预测结果
        predicted_labels = [self.id_to_label[pred.item()] for pred in predictions[0]]
        
        # 6. 提取实体
        entities = self.extract_entities(words, predicted_labels)
        
        # 7. 后处理和格式化
        result = self.post_process_entities(entities)
        
        return result
    
    def extract_entities(self, words, labels):
        """从预测标签中提取实体"""
        entities = {}
        current_entity = None
        current_text = []
        
        for word, label in zip(words, labels):
            if label.startswith('B-'):
                # 保存之前的实体
                if current_entity and current_text:
                    entities[current_entity] = ' '.join(current_text)
                
                # 开始新实体
                current_entity = label[2:]
                current_text = [word]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # 继续当前实体
                current_text.append(word)
                
            else:
                # 结束当前实体
                if current_entity and current_text:
                    entities[current_entity] = ' '.join(current_text)
                current_entity = None
                current_text = []
        
        # 处理最后一个实体
        if current_entity and current_text:
            entities[current_entity] = ' '.join(current_text)
            
        return entities
    
    def post_process_entities(self, entities):
        """后处理提取的实体"""
        result = {
            "InvoiceNo": "",
            "InvoiceDate": "",
            "Currency": "",
            "Amount with Tax": "",
            "Amount without Tax": "",
            "Tax": ""
        }
        
        # 发票号码处理
        if 'InvoiceNo' in entities:
            result['InvoiceNo'] = entities['InvoiceNo'].strip()
        
        # 日期处理
        if 'InvoiceDate' in entities:
            date_text = entities['InvoiceDate']
            result['InvoiceDate'] = self.normalize_date(date_text)
        
        # 货币处理
        if 'Currency' in entities:
            result['Currency'] = entities['Currency'].strip().upper()
        
        # 金额处理
        if 'AmountWithTax' in entities:
            result['Amount with Tax'] = self.normalize_amount(entities['AmountWithTax'])
        
        if 'AmountWithoutTax' in entities:
            result['Amount without Tax'] = self.normalize_amount(entities['AmountWithoutTax'])
        
        if 'Tax' in entities:
            result['Tax'] = self.normalize_amount(entities['Tax'])
        
        return result
    
    def normalize_date(self, date_text):
        """标准化日期格式"""
        # 中文日期模式
        chinese_pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})日'
        match = re.search(chinese_pattern, date_text)
        if match:
            year, month, day = match.groups()
            return f"{year}年{month.zfill(2)}月{day.zfill(2)}日"
        
        return date_text.strip()
    
    def normalize_amount(self, amount_text):
        """标准化金额格式"""
        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', amount_text)
        if numbers:
            return numbers[0]
        return "0"