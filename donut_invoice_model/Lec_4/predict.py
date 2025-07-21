import json
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import argparse
from pathlib import Path
import logging
from postprocess import infer_fields, clean_json_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DonutInvoicePredictor:
    """Donut 发票信息抽取预测器"""
    
    def __init__(self, model_path="./donut-invoice-finetuned"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和处理器
        logger.info(f"从 {model_path} 加载模型...")
        self.processor = DonutProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型加载完成，使用设备: {self.device}")
    
    def predict(self, image_path, prompt="<s_invoice>"):
        """对单张发票图片进行预测"""
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 预处理图片
        pixel_values = self.processor(
            image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # 编码提示词
        prompt_ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # 生成预测
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                decoder_input_ids=prompt_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
        
        # 解码结果
        generated_text = self.processor.batch_decode(
            generated_ids.sequences, 
            skip_special_tokens=True
        )[0]
        
        # 清理和解析 JSON
        try:
            # 移除提示词
            json_string = generated_text.replace(prompt, "").strip()
            json_string = clean_json_string(json_string)
            
            # 解析 JSON
            result = json.loads(json_string)
            
            # 后处理推理字段
            result = infer_fields(result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析错误: {e}")
            logger.error(f"生成的文本: {generated_text}")
            return {"error": "JSON解析失败", "raw_text": generated_text}
    
    def batch_predict(self, image_dir, output_dir=None):
        """批量预测"""
        image_dir = Path(image_dir)
        results = {}
        
        for img_file in image_dir.glob("*.jpg"):
            logger.info(f"处理图片: {img_file.name}")
            result = self.predict(img_file)
            results[img_file.name] = result
            
            # 保存单个结果
            if output_dir:
                output_path = Path(output_dir) / f"{img_file.stem}_result.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Donut 发票信息抽取")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, help="图片目录路径")
    parser.add_argument("--model_path", type=str, default="./donut-invoice-finetuned", help="模型路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--output_dir", type=str, help="批量输出目录")
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = DonutInvoicePredictor(args.model_path)
    
    if args.image:
        # 单张图片预测
        result = predictor.predict(args.image)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.image_dir:
        # 批量预测
        results = predictor.batch_predict(args.image_dir, args.output_dir)
        logger.info(f"批量处理完成，共处理 {len(results)} 张图片")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()