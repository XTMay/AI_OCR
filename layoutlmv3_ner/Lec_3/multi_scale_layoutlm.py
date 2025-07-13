import torch
import torch.nn as nn
from transformers import LayoutLMv3Model, LayoutLMv3Config

class MultiScaleLayoutLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 基础LayoutLM模型
        self.layoutlm = LayoutLMv3Model(config)
        
        # 多尺度特征提取器
        self.multi_scale_extractor = MultiScaleFeatureExtractor(
            hidden_size=config.hidden_size
        )
        
        # 注意力融合模块
        self.attention_fusion = AttentionFusion(
            hidden_size=config.hidden_size,
            num_heads=8
        )
        
        # 分类头
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, input_ids, bbox, attention_mask=None, 
                token_type_ids=None, position_ids=None, 
                head_mask=None, labels=None):
        
        # LayoutLM编码
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        # 多尺度特征提取
        multi_scale_features = self.multi_scale_extractor(sequence_output)
        
        # 特征融合
        fused_features = self.attention_fusion(
            sequence_output, 
            multi_scale_features
        )
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), 
                           labels.view(-1))
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        x = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        
        multi_scale_outputs = []
        for conv in self.conv_layers:
            output = torch.relu(conv(x))
            multi_scale_outputs.append(output)
        
        # 拼接多尺度特征
        concatenated = torch.cat(multi_scale_outputs, dim=1)
        return concatenated.transpose(1, 2)

class AttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, base_features, multi_scale_features):
        # 注意力融合
        attn_output, _ = self.multihead_attn(
            query=base_features,
            key=multi_scale_features,
            value=multi_scale_features
        )
        
        # 残差连接和层归一化
        fused = self.layer_norm(base_features + attn_output)
        return fused