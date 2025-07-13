import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer
import json
from PIL import Image

class InvoiceLayoutDataset(Dataset):
    def __init__(self, annotations_file, tokenizer, max_seq_length=512):
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # 定义标签映射
        self.label_map = {
            'O': 0,
            'B-InvoiceNo': 1,
            'I-InvoiceNo': 2,
            'B-InvoiceDate': 3,
            'I-InvoiceDate': 4,
            'B-Currency': 5,
            'I-Currency': 6,
            'B-AmountWithTax': 7,
            'I-AmountWithTax': 8,
            'B-AmountWithoutTax': 9,
            'I-AmountWithoutTax': 10,
            'B-Tax': 11,
            'I-Tax': 12
        }
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 加载图像
        image = Image.open(annotation['image']).convert('RGB')
        
        # 提取文本和边界框
        words = []
        boxes = []
        labels = []
        
        for entity in annotation['entities']:
            word_tokens = self.tokenizer.tokenize(entity['text'])
            words.extend(word_tokens)
            
            # 为每个token分配相同的边界框
            for i, token in enumerate(word_tokens):
                boxes.append(entity['bbox'])
                # BIO标注
                if i == 0:
                    labels.append(self.label_map[f"B-{entity['label']}"])
                else:
                    labels.append(self.label_map[f"I-{entity['label']}"])
        
        # 编码
        encoding = self.tokenizer(
            words,
            boxes=boxes,
            word_labels=labels,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bbox': encoding['bbox'].flatten(),
            'labels': encoding['labels'].flatten()
        }