import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from PIL import Image
import random
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceDataset(Dataset):
    """多语言发票数据集"""
    
    def __init__(self, dataset_path, processor, max_length=512, split="train"):
        self.dataset_path = Path(dataset_path)
        self.processor = processor
        self.max_length = max_length
        self.split = split
        
        # 加载图片和标注文件
        self.image_dir = self.dataset_path / "images"
        self.annotation_dir = self.dataset_path / "annotations"
        
        # 获取所有样本
        self.samples = []
        for img_file in self.image_dir.glob("*.jpg"):
            json_file = self.annotation_dir / f"{img_file.stem}.json"
            if json_file.exists():
                self.samples.append({
                    "image_path": img_file,
                    "json_path": json_file
                })
        
        logger.info(f"加载了 {len(self.samples)} 个 {split} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # 加载标注
        with open(sample["json_path"], "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        
        # 构建提示词和目标序列
        prompt = "<s_invoice>"
        target_sequence = json.dumps(ground_truth, ensure_ascii=False, separators=(',', ':'))
        target_sequence = f"{prompt}{target_sequence}</s>"
        
        # 处理图片和文本
        pixel_values = self.processor(
            image, 
            return_tensors="pt"
        ).pixel_values.squeeze()
        
        # 编码目标序列
        target_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {
            "pixel_values": pixel_values,
            "labels": target_ids
        }

def collate_fn(batch):
    """数据批处理函数"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

# 在 main() 函数的最后部分，修改保存配置的代码
def main():
    # 配置参数
    config = {
        "model_name": "naver-clova-ix/donut-base-finetuned-docvqa",
        "dataset_path": "./dataset",
        "output_dir": "./donut-invoice-finetuned",
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "warmup_steps": 300,
        "max_length": 512,
        "image_size": [1280, 960]
    }
    
    # 加载处理器和模型
    logger.info(f"加载模型: {config['model_name']}")
    processor = DonutProcessor.from_pretrained(config["model_name"])
    model = VisionEncoderDecoderModel.from_pretrained(config["model_name"])
    
    # 更新处理器配置
    processor.image_processor.size = config["image_size"]
    processor.image_processor.do_align_long_axis = False
    
    # 添加新的特殊标记
    special_tokens = ["<s_invoice>", "</s_invoice>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # 设置生成配置
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_invoice>"])[0]
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = config["max_length"]
    # 修复：要么设置 num_beams > 1，要么禁用 early_stopping
    model.config.early_stopping = True  # 改为 False
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 1
    
    # 创建数据集
    train_dataset = InvoiceDataset(
        config["dataset_path"], 
        processor, 
        config["max_length"], 
        "train"
    )
    
    # 检查数据集大小
    if len(train_dataset) == 0:
        logger.error("训练数据集为空！请检查 dataset/images/ 和 dataset/annotations/ 目录")
        logger.error("确保图片文件（.jpg）和对应的标注文件（.json）存在且文件名匹配")
        return
    
    # 如果数据集太小，调整训练参数
    if len(train_dataset) < 10:
        logger.warning(f"训练数据集较小（{len(train_dataset)} 个样本），调整训练参数")
        config["num_epochs"] = 3  # 减少训练轮数
        config["save_steps"] = max(1, len(train_dataset))  # 调整保存步数
        config["eval_steps"] = max(1, len(train_dataset))  # 调整评估步数
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        warmup_steps=config["warmup_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",  # 修改：evaluation_strategy -> eval_strategy
        eval_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # 简化示例，实际应使用验证集
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到: {config['output_dir']}")
    trainer.save_model()
    processor.save_pretrained(config["output_dir"])
    
    # 修改：保存训练配置到不同的文件名，避免覆盖模型配置
    with open(os.path.join(config["output_dir"], "training_config.json"), "w") as f:  # 改为 training_config.json
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main()