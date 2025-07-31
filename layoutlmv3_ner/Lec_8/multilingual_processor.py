import unicodedata
import re
from transformers import AutoTokenizer
from langdetect import detect

class MultilingualProcessor:
    def __init__(self):
        self.language_configs = {
            'zh': {
                'tokenizer': 'bert-base-chinese',
                'direction': 'ltr',
                'script': 'han',
                'date_patterns': [
                    r'(\d{4})年(\d{1,2})月(\d{1,2})日',
                    r'(\d{4})-(\d{1,2})-(\d{1,2})'
                ]
            },
            'en': {
                'tokenizer': 'bert-base-uncased',
                'direction': 'ltr',
                'script': 'latin',
                'date_patterns': [
                    r'(\d{1,2})/(\d{1,2})/(\d{4})',
                    r'(\w+)\s+(\d{1,2}),\s+(\d{4})'
                ]
            },
            'ja': {
                'tokenizer': 'cl-tohoku/bert-base-japanese',
                'direction': 'ltr',
                'script': 'mixed',
                'date_patterns': [
                    r'(\d{4})年(\d{1,2})月(\d{1,2})日',
                    r'令和(\d+)年(\d{1,2})月(\d{1,2})日'
                ]
            },
            'ar': {
                'tokenizer': 'asafaya/bert-base-arabic',
                'direction': 'rtl',
                'script': 'arabic',
                'date_patterns': [
                    r'(\d{1,2})/(\d{1,2})/(\d{4})',
                    r'(\d{4})-(\d{1,2})-(\d{1,2})'
                ]
            }
        }
    
    def detect_language(self, text):
        """检测文本语言"""
        try:
            detected_lang = detect(text)
            return detected_lang if detected_lang in self.language_configs else 'en'
        except:
            return 'en'
    
    def normalize_text(self, text, language):
        """文本标准化"""
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 语言特定处理
        if language == 'zh':
            # 中文特定处理
            text = self.process_chinese_text(text)
        elif language == 'ja':
            # 日文特定处理
            text = self.process_japanese_text(text)
        elif language == 'ar':
            # 阿拉伯文特定处理
            text = self.process_arabic_text(text)
        
        return text
    
    def process_chinese_text(self, text):
        """中文文本处理"""
        # 繁简转换
        # 数字标准化
        chinese_numbers = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10', '百': '100', '千': '1000', '万': '10000'
        }
        
        for cn_num, ar_num in chinese_numbers.items():
            text = text.replace(cn_num, ar_num)
        
        return text
    
    def extract_multilingual_fields(self, text, language):
        """多语言字段提取"""
        config = self.language_configs.get(language, self.language_configs['en'])
        
        extracted_fields = {}
        
        # 日期提取
        for pattern in config['date_patterns']:
            matches = re.findall(pattern, text)
            if matches:
                extracted_fields['date'] = matches[0]
                break
        
        # 金额提取（通用模式）
        amount_patterns = [
            r'[¥$€£]\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*[¥$€£]',
            r'([\d,]+\.?\d*)\s*(元|USD|EUR|GBP)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            if matches:
                extracted_fields['amount'] = matches[0]
                break
        
        return extracted_fields

class MultilingualLayoutLM:
    def __init__(self):
        self.language_models = {}
        self.processor = MultilingualProcessor()
    
    def load_language_model(self, language):
        """加载特定语言模型"""
        if language not in self.language_models:
            config = self.processor.language_configs[language]
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            
            # 加载对应的LayoutLM模型
            model = self.load_layoutlm_for_language(language)
            
            self.language_models[language] = {
                'tokenizer': tokenizer,
                'model': model
            }
    
    def predict_multilingual(self, image_path, text_blocks):
        """多语言预测"""
        # 检测主要语言
        all_text = ' '.join([block['text'] for block in text_blocks])
        detected_language = self.processor.detect_language(all_text)
        
        # 加载对应语言模型
        self.load_language_model(detected_language)
        
        # 使用对应模型进行预测
        model_info = self.language_models[detected_language]
        
        # 预处理
        processed_blocks = []
        for block in text_blocks:
            normalized_text = self.processor.normalize_text(
                block['text'], 
                detected_language
            )
            processed_blocks.append({
                **block,
                'text': normalized_text
            })
        
        # 模型推理
        result = self.inference_with_language_model(
            model_info, 
            processed_blocks, 
            detected_language
        )
        
        return result