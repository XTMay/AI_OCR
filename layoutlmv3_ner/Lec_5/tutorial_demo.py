import os
import json
import logging
from datetime import datetime
from src.data_processor import InvoiceDataProcessor
from src.layoutlm_trainer import LayoutLMv3Trainer, InvoiceLayoutDataset
from src.inference_pipeline import InvoiceInferencePipeline
from src.training_monitor import TrainingMonitor
from src.performance_diagnostics import PerformanceDiagnostics
from src.progressive_training import ProgressiveTrainingStrategy

class InvoiceOCRTutorial:
    """发票OCR完整教学演示"""
    
    def __init__(self):
        self.setup_logging()
        self.config = self.load_config()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'tutorial_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> dict:
        """加载配置"""
        base_dir = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5"
        
        # 生成所有PDF文件路径
        pdf_files = []
        label_files = []
        
        for i in range(1, 8):  # 1到7
            pdf_path = f"{base_dir}/data/raw/測試股份有限公司_{i}.pdf"
            label_path = f"{base_dir}/data/training/label_{i}.json"
            
            pdf_files.append(pdf_path)
            label_files.append(label_path)
        
        return {
            "data_dir": f"{base_dir}/data",
            "pdf_files": pdf_files,
            "label_files": label_files,
            "model_output_dir": "./models/tutorial_invoice_layoutlmv3",
            "final_model_path": "./models/final_tutorial_invoice_layoutlmv3"
        }
    
    def run_complete_tutorial(self):
        """运行完整教学演示"""
        self.logger.info("🎓 开始发票OCR完整教学演示")
        
        try:
            # 验证数据文件存在性
            self.validate_data_files()
            
            # 第一步：数据预处理演示
            self.logger.info("\n" + "="*50)
            self.logger.info("📊 第一步：数据预处理演示")
            self.logger.info("="*50)
            
            processor = self.demonstrate_data_preprocessing()
            
            # 第二步：数据标注演示
            self.logger.info("\n" + "="*50)
            self.logger.info("🏷️ 第二步：数据标注演示")
            self.logger.info("="*50)
            
            annotations = self.demonstrate_annotation_process(processor)
            
            # 第三步：模型训练演示
            self.logger.info("\n" + "="*50)
            self.logger.info("🚀 第三步：模型训练演示")
            self.logger.info("="*50)
            
            model_path = self.demonstrate_model_training(annotations)
            
            # 第四步：推理演示
            self.logger.info("\n" + "="*50)
            self.logger.info("🔮 第四步：推理演示")
            self.logger.info("="*50)
            
            self.demonstrate_inference(model_path)
            
            # 第五步：性能分析
            self.logger.info("\n" + "="*50)
            self.logger.info("📈 第五步：性能分析")
            self.logger.info("="*50)
            
            self.demonstrate_performance_analysis()
            
            self.logger.info("\n🎉 教学演示完成！")
            
        except Exception as e:
            self.logger.error(f"教学演示失败: {e}")
            raise
    
    def validate_data_files(self):
        """验证数据文件存在性"""
        self.logger.info("🔍 验证数据文件...")
        
        missing_files = []
        
        # 检查PDF文件
        for pdf_file in self.config["pdf_files"]:
            if not os.path.exists(pdf_file):
                missing_files.append(pdf_file)
        
        # 检查标注文件
        for label_file in self.config["label_files"]:
            if not os.path.exists(label_file):
                missing_files.append(label_file)
        
        if missing_files:
            self.logger.error("❌ 以下文件不存在:")
            for file in missing_files:
                self.logger.error(f"  - {file}")
            raise FileNotFoundError(f"缺少 {len(missing_files)} 个数据文件")
        
        self.logger.info(f"✅ 所有数据文件验证通过 ({len(self.config['pdf_files'])} PDF + {len(self.config['label_files'])} 标注)")
    
    def demonstrate_data_preprocessing(self) -> InvoiceDataProcessor:
        """演示数据预处理"""
        self.logger.info("初始化数据处理器...")
        processor = InvoiceDataProcessor(self.config["data_dir"])
        
        # 处理所有PDF文件
        all_image_paths = []
        
        self.logger.info("📄 批量PDF转图像演示:")
        for i, pdf_path in enumerate(self.config["pdf_files"], 1):
            self.logger.info(f"  处理PDF {i}/11: {os.path.basename(pdf_path)}")
            try:
                image_paths = processor.pdf_to_images(pdf_path)
                all_image_paths.extend(image_paths)
                self.logger.info(f"    ✅ 转换了 {len(image_paths)} 张图像")
            except Exception as e:
                self.logger.error(f"    ❌ 转换失败: {e}")
        
        self.logger.info(f"  📸 总计转换 {len(all_image_paths)} 张图像")
        
        # OCR演示（使用第一个图像）
        self.logger.info("\n🔍 OCR文本提取演示:")
        if all_image_paths:
            # 修复：使用正确的方法名
            ocr_data = processor.extract_text_with_positions(all_image_paths[0])
            self.logger.info(f"  ✅ 提取到 {len(ocr_data)} 个文本块")
            
            # 显示前几个文本块作为示例
            for i, item in enumerate(ocr_data[:3]):
                self.logger.info(f"    📝 文本块 {i+1}: '{item['text'][:20]}...' 置信度: {item['confidence']:.2f}")
        
        # 加载所有标注数据
        self.logger.info("\n🏷️ 批量标注数据加载演示:")
        all_labels = []
        for i, label_path in enumerate(self.config["label_files"], 1):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                all_labels.append(label_data)
                self.logger.info(f"  ✅ 标注文件 {i}: {len(label_data)} 个字段")
            except Exception as e:
                self.logger.error(f"  ❌ 加载标注文件 {i} 失败: {e}")
        
        # 显示第一个标注文件的内容作为示例
        if all_labels:
            self.logger.info("\n  📋 标注字段示例 (文件1):")
            for field, value in list(all_labels[0].items())[:5]:  # 只显示前5个字段
                self.logger.info(f"    🔖 {field}: {value}")
        
        return processor
    
    def demonstrate_annotation_process(self, processor: InvoiceDataProcessor):
        """演示标注过程"""
        self.logger.info("🏷️ 批量创建训练标注...")
        
        all_annotations = []
        
        # 处理所有文件对
        for i, (pdf_path, label_path) in enumerate(zip(self.config["pdf_files"], self.config["label_files"]), 1):
            try:
                self.logger.info(f"  处理文件对 {i}/11...")
                
                # 加载标注数据
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                
                # 获取图像路径
                image_paths = processor.pdf_to_images(pdf_path)
                if not image_paths:
                    self.logger.warning(f"    ⚠️ 文件 {i} 没有可用的图像")
                    continue
                
                # 创建训练标注
                # 修复：调整参数顺序，传入图像路径列表而不是单个图像
                annotations = processor.create_training_annotations([image_paths[0]], label_data)
                all_annotations.extend(annotations)
                
                self.logger.info(f"    ✅ 创建了 {len(annotations)} 个标注")
                
            except Exception as e:
                self.logger.error(f"    ❌ 处理文件对 {i} 失败: {e}")
        
        self.logger.info(f"\n  📊 总计创建 {len(all_annotations)} 个标注")
        
        # 显示标注统计
        if all_annotations:
            label_counts = {}
            for ann in all_annotations:
                label = ann['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            self.logger.info("  📈 标注分布:")
            for label, count in label_counts.items():
                self.logger.info(f"    🏷️ {label}: {count} 个")
        
        # 保存标注结果到指定文件夹
        self._save_annotations_to_folder(all_annotations)
        
        return all_annotations
    
    def _save_annotations_to_folder(self, annotations):
        """保存标注结果到annotations文件夹"""
        try:
            # 创建annotations文件夹
            annotations_dir = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation"
            os.makedirs(annotations_dir, exist_ok=True)
            
            # 保存完整的标注数据
            annotations_file = os.path.join(annotations_dir, f"all_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # 转换数据为可序列化格式
            serializable_annotations = []
            for ann in annotations:
                try:
                    # 安全处理bbox数据
                    bbox = ann['bbox']
                    if isinstance(bbox, (list, tuple)):
                        # 确保bbox中的所有值都是有效的数字
                        safe_bbox = []
                        for coord in bbox:
                            if isinstance(coord, (int, float)) and not (coord == float('inf') or coord == float('-inf') or coord != coord):  # 检查nan
                                safe_bbox.append(float(coord))
                            else:
                                safe_bbox.append(0.0)  # 使用默认值替换无效坐标
                    else:
                        safe_bbox = [0.0, 0.0, 100.0, 20.0]  # 默认bbox
                    
                    serializable_ann = {
                        'image_path': str(ann['image_path']),
                        'text': str(ann['text']),
                        'label': str(ann['label']),
                        'bbox': safe_bbox,
                        'confidence': float(ann.get('confidence', 1.0))
                    }
                    serializable_annotations.append(serializable_ann)
                    
                except Exception as e:
                    self.logger.warning(f"  ⚠️ 跳过无效标注: {e}")
                    continue
            
            # 保存JSON文件
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_annotations, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"  💾 标注结果已保存到: {annotations_file}")
            self.logger.info(f"  📊 成功保存 {len(serializable_annotations)} 个标注")
            
            # 按标签分类保存
            label_groups = {}
            for ann in serializable_annotations:
                label = ann['label']
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(ann)
            
            # 为每个标签创建单独的文件
            for label, label_annotations in label_groups.items():
                label_file = os.path.join(annotations_dir, f"{label}_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(label_annotations, f, ensure_ascii=False, indent=2)
                self.logger.info(f"  📁 {label} 标注已保存到: {label_file}")
            
            # 创建统计摘要
            summary = {
                "总标注数量": len(serializable_annotations),
                "标注分布": {label: len(anns) for label, anns in label_groups.items()},
                "创建时间": datetime.now().isoformat(),
                "文件列表": {
                    "完整标注": annotations_file,
                    "分类标注": {label: f"{label}_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" for label in label_groups.keys()}
                }
            }
            
            summary_file = os.path.join(annotations_dir, f"annotation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"  📋 标注摘要已保存到: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"  ❌ 保存标注结果失败: {e}")
    
    def demonstrate_model_training(self, annotations):
        """演示模型训练"""
        self.logger.info("🚀 开始模型训练演示...")
        
        if not annotations:
            self.logger.warning("⚠️ 没有可用的标注数据，跳过训练")
            return None
        
        try:
            # 初始化训练器
            trainer = LayoutLMv3Trainer()
            
            # 创建数据集
            train_data = []
            for ann in annotations:
                # Convert numpy types to Python native types for JSON serialization
                bbox = ann['bbox']
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                elif isinstance(bbox, (list, tuple)):
                    bbox = [int(x) if hasattr(x, 'item') else x for x in bbox]
                
                train_data.append({
                    'image_path': str(ann['image_path']),
                    'entities': [{
                        'text': str(ann['text']),
                        'bbox': bbox,
                        'label': str(ann['label'])
                    }]
                })
            
            # 保存训练数据
            os.makedirs("data/training", exist_ok=True)
            with open("data/training/demo_train.json", 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            # 创建数据集
            train_dataset = InvoiceLayoutDataset("data/training/demo_train.json", trainer.tokenizer)
            val_dataset = train_dataset  # 演示用，实际应该分开
            # train_dataset， val_dataset， test_dataset

            
            self.logger.info(f"  ✅ 训练数据集大小: {len(train_dataset)}")
            
            # 设置训练（演示模式，少量epoch）
            trainer.setup_training(
                train_dataset, 
                val_dataset, 
                self.config["model_output_dir"],
                num_epochs=2  # 增加到2个epoch以获得更好效果
            )
            
            # 开始训练
            self.logger.info("  🏃‍♂️ 开始训练（演示模式）...")
            trainer.train()
            
            # 保存模型
            trainer.save_model(self.config["final_model_path"])
            self.logger.info(f"  ✅ 模型保存到: {self.config['final_model_path']}")
            
            return self.config["final_model_path"]
            
        except Exception as e:
            self.logger.error(f"  ❌ 训练失败: {e}")
            return None
    
    def demonstrate_inference(self, model_path):
        """演示推理过程"""
        self.logger.info("🔮 开始批量推理演示...")
        
        if not model_path or not os.path.exists(model_path):
            self.logger.warning("⚠️ 模型不存在，跳过推理演示")
            return
        
        try:
            # 初始化推理管道
            inference_pipeline = InvoiceInferencePipeline(model_path)
            
            total_accuracy = 0
            successful_inferences = 0
            
            # 创建推理结果保存目录
            inference_results_dir = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/inference_results"
            os.makedirs(inference_results_dir, exist_ok=True)
            
            # 处理所有发票
            for i, (pdf_path, label_path) in enumerate(zip(self.config["pdf_files"], self.config["label_files"]), 1):
                self.logger.info(f"\n  📄 处理发票 {i}/11: {os.path.basename(pdf_path)}")
                
                try:
                    # 推理
                    invoice_info = inference_pipeline.process_invoice(pdf_path)
                    
                    # 保存推理结果到JSON文件
                    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
                    inference_result_file = os.path.join(inference_results_dir, f"inference_{pdf_name}.json")
                    
                    with open(inference_result_file, 'w', encoding='utf-8') as f:
                        json.dump(invoice_info, f, ensure_ascii=False, indent=2)
                    
                    self.logger.info(f"    💾 推理结果已保存到: {inference_result_file}")
                    
                    if invoice_info and isinstance(invoice_info, dict):
                        # 加载ground truth
                        with open(label_path, 'r', encoding='utf-8') as f:
                            ground_truth = json.load(f)
                        
                        # 创建对比结果
                        comparison_result = {
                            "pdf_file": pdf_name,
                            "inference_results": invoice_info,
                            "ground_truth": ground_truth,
                            "field_comparison": {},
                            "accuracy_metrics": {}
                        }
                        
                        # 计算准确性
                        correct_fields = 0
                        total_fields = len(ground_truth)
                        
                        for field, true_value in ground_truth.items():
                            pred_value = invoice_info.get(field, "")
                            is_correct = str(pred_value).strip() == str(true_value).strip()
                            
                            comparison_result["field_comparison"][field] = {
                                "predicted": str(pred_value).strip(),
                                "ground_truth": str(true_value).strip(),
                                "correct": is_correct
                            }
                            
                            if is_correct:
                                correct_fields += 1
                        
                        accuracy = correct_fields / total_fields if total_fields > 0 else 0
                        comparison_result["accuracy_metrics"] = {
                            "correct_fields": correct_fields,
                            "total_fields": total_fields,
                            "accuracy": accuracy
                        }
                        
                        # 保存对比结果
                        comparison_file = os.path.join(inference_results_dir, f"comparison_{pdf_name}.json")
                        with open(comparison_file, 'w', encoding='utf-8') as f:
                            json.dump(comparison_result, f, ensure_ascii=False, indent=2)
                        
                        self.logger.info(f"    📊 对比结果已保存到: {comparison_file}")
                        
                        total_accuracy += accuracy
                        successful_inferences += 1
                        
                        self.logger.info(f"    ✅ 推理成功，准确率: {accuracy:.2%} ({correct_fields}/{total_fields})")
                        
                    else:
                        self.logger.error(f"    ❌ 推理失败: {invoice_info}")
                        
                        # 保存失败结果
                        failure_result = {
                            "pdf_file": pdf_name,
                            "status": "failed",
                            "result": invoice_info,
                            "error": "推理返回了无效结果"
                        }
                        
                        failure_file = os.path.join(inference_results_dir, f"failure_{pdf_name}.json")
                        with open(failure_file, 'w', encoding='utf-8') as f:
                            json.dump(failure_result, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    self.logger.error(f"    ❌ 处理失败: {e}")
            
            # 计算平均准确率
            if successful_inferences > 0:
                avg_accuracy = total_accuracy / successful_inferences
                self.logger.info(f"\n  🎯 平均准确率: {avg_accuracy:.2%} (成功处理 {successful_inferences}/7 个文件)")
                
                # 创建总结报告
                summary_report = {
                    "inference_summary": {
                        "total_files": len(self.config["pdf_files"]),
                        "successful_inferences": successful_inferences,
                        "failed_inferences": len(self.config["pdf_files"]) - successful_inferences,
                        "average_accuracy": avg_accuracy,
                        "total_accuracy": total_accuracy
                    },
                    "file_details": [],
                    "inference_results_directory": inference_results_dir,
                    "ground_truth_directory": "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/training",
                    "timestamp": datetime.now().isoformat()
                }
                
                # 添加每个文件的详细信息
                for i, (pdf_path, label_path) in enumerate(zip(self.config["pdf_files"], self.config["label_files"]), 1):
                    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
                    summary_report["file_details"].append({
                        "file_number": i,
                        "pdf_file": pdf_name,
                        "inference_result_file": f"inference_{pdf_name}.json",
                        "comparison_file": f"comparison_{pdf_name}.json",
                        "ground_truth_file": os.path.basename(label_path)
                    })
                
                # 保存总结报告
                summary_file = os.path.join(inference_results_dir, "inference_summary_report.json")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_report, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"  📋 总结报告已保存到: {summary_file}")
                self.logger.info(f"  📁 所有推理结果保存在: {inference_results_dir}")
                
            else:
                self.logger.warning("  ⚠️ 没有成功的推理结果")
                
        except Exception as e:
            self.logger.error(f"  ❌ 推理演示失败: {e}")
    
    def demonstrate_performance_analysis(self):
        """演示性能分析"""
        self.logger.info("📈 性能分析演示...")
        
        try:
            # 初始化性能诊断器
            diagnostics = PerformanceDiagnostics()
            
            # 分析第一个PDF的OCR性能作为示例
            self.logger.info("  🔍 OCR性能分析 (示例文件):")
            ocr_metrics = diagnostics.analyze_ocr_performance(self.config["pdf_files"][0])
            
            for metric, value in ocr_metrics.items():
                self.logger.info(f"    📊 {metric}: {value}")
            # ROC-AUC, epcho/bach-loss
            
            # 提供优化建议
            self.logger.info("\n  💡 基于多文件数据的优化建议:")
            suggestions = [
                "利用11个样本的多样性进行数据增强",
                "分析不同发票格式的共同特征和差异",
                "使用交叉验证评估模型在不同样本上的表现",
                "实施集成学习以提高对格式变化的鲁棒性",
                "建立持续学习机制以适应新的发票类型",
                "优化OCR预处理以处理不同质量的扫描件",
                "使用主动学习策略选择最有价值的标注样本"
            ]
            
            for i, suggestion in enumerate(suggestions, 1):
                self.logger.info(f"    {i}. {suggestion}")
                
        except Exception as e:
            self.logger.error(f"  ❌ 性能分析失败: {e}")


if __name__ == '__main__':
    import multiprocessing
    # 设置multiprocessing启动方法，防止semaphore泄漏
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误
    
    tutorial = InvoiceOCRTutorial()
    tutorial.run_complete_tutorial()