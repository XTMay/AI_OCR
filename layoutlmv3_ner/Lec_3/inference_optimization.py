import torch
import onnx
import onnxruntime as ort
from torch.jit import script

class InferenceOptimizer:
    def __init__(self, model):
        self.model = model
    
    def convert_to_torchscript(self, save_path):
        """转换为TorchScript"""
        self.model.eval()
        
        # 示例输入
        example_input = {
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'bbox': torch.randint(0, 1000, (1, 512, 4)),
            'attention_mask': torch.ones(1, 512)
        }
        
        # 转换为TorchScript
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(save_path)
        
        return traced_model
    
    def convert_to_onnx(self, save_path):
        """转换为ONNX格式"""
        self.model.eval()
        
        # 示例输入
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'bbox': torch.randint(0, 1000, (1, 512, 4)),
            'attention_mask': torch.ones(1, 512)
        }
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'bbox', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'bbox': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )
    
    def quantize_model(self):
        """模型量化"""
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        return quantized_model

class BatchInference:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
    
    def batch_predict(self, images):
        """批量推理"""
        results = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # 批量处理
            batch_inputs = self.prepare_batch_inputs(batch_images)
            
            with torch.no_grad():
                batch_outputs = self.model(**batch_inputs)
            
            # 解析批量结果
            batch_results = self.parse_batch_outputs(batch_outputs)
            results.extend(batch_results)
        
        return results