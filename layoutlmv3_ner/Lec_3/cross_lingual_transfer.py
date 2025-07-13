import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class CrossLingualLayoutLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 使用多语言预训练模型作为骨干
        self.xlm_roberta = XLMRobertaModel.from_pretrained(
            'xlm-roberta-base'
        )
        
        # 位置编码层
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        
        # 语言特定适配器
        self.language_adapters = nn.ModuleDict({
            lang: LanguageAdapter(config.hidden_size)
            for lang in ['zh', 'en', 'ja', 'ko', 'ar', 'es', 'fr', 'de']
        })
        
        # 跨语言对齐层
        self.cross_lingual_alignment = CrossLingualAlignment(
            config.hidden_size
        )
        
        # 分类头
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, bbox, attention_mask, language_id):
        # XLM-RoBERTa编码
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        # 添加位置信息
        position_embeddings = self.get_position_embeddings(bbox)
        sequence_output = sequence_output + position_embeddings
        
        # 语言特定适配
        if language_id in self.language_adapters:
            sequence_output = self.language_adapters[language_id](
                sequence_output
            )
        
        # 跨语言对齐
        aligned_output = self.cross_lingual_alignment(sequence_output)
        
        # 分类预测
        logits = self.classifier(aligned_output)
        
        return {'logits': logits}
    
    def get_position_embeddings(self, bbox):
        """获取位置嵌入"""
        # 将边界框转换为位置嵌入
        # 这里简化处理，实际应该更复杂
        position_ids = torch.arange(
            bbox.size(1), 
            device=bbox.device
        ).unsqueeze(0).expand(bbox.size(0), -1)
        
        return self.position_embeddings(position_ids)

class LanguageAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = self.layer_norm(x + residual)
        return x

class CrossLingualAlignment(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.alignment_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # 自注意力对齐
        aligned, _ = self.alignment_layer(x, x, x)
        return self.layer_norm(x + aligned)