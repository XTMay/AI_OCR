# 🧠 AI OCR Course: Hands-On Document Intelligence with PaddleOCR, LayoutLMv3, and Donut

Welcome to this open, hands-on OCR and Document Understanding course!  
You'll explore and compare three popular models: **PaddleOCR**, **LayoutLMv3**, and **Donut**,  
implementing real-world OCR pipelines with **images**, **PDFs**, and **structured outputs** like JSON.

---

## 📚 Course Objectives

By the end of this course, you will be able to:

- ✅ Understand the document understanding pipeline: OCR → Structure Extraction → Output Parsing
- ✅ Apply PaddleOCR for high-accuracy multilingual text detection and recognition
- ✅ Fine-tune LayoutLMv3 for named entity recognition (NER) using OCR + coordinates
- ✅ Use Donut for OCR-free, end-to-end document parsing (image → JSON)
- ✅ Work with image, PDF, and scanned document inputs

---

## 📦 Models Covered
``
| Model        | Type                         | Strengths                                     | Usage |
|--------------|------------------------------|-----------------------------------------------|-------|
| 🥇 PaddleOCR | Traditional OCR + Structure  | Fast, accurate, great for Chinese/English OCR | Text + layout extraction |
| 🥈 LayoutLMv3 | Multimodal Transformer       | Strong in field-level NER, form understanding | Requires OCR output as input |
| 🥉 Donut     | OCR-free Image2Text Transformer | Direct image → JSON, template-robust        | Needs fine-tuning for custom formats |
``
---

## 🖼️ Supported Input Types

- 🖼️ JPEG / PNG / TIFF scanned images
- 📄 PDF documents (auto-splitted into pages)
- 📸 Screenshots
- 🧾 Printed or handwritten invoices, forms, receipts

---

## 📂 Project Structure

ai-ocr-course/
├── paddleocr_demo/          # PaddleOCR-based OCR pipeline
├── layoutlmv3_ner/          # LayoutLMv3 for field/entity extraction
├── donut_inference/         # Donut model inference and fine-tuning
├── utils/                   # PDF/image processing utilities
├── data/                    # Sample documents and annotations
└── README.md

---

## 🚀 Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/your-org/ai-ocr-course.git
cd ai-ocr-course

2. Install Dependencies

pip install -r requirements.txt

Or install each submodule dependencies individually in their respective folders.

3. Try a demo

PaddleOCR

cd paddleocr_demo
python run_ocr.py --image sample_invoice.jpg

LayoutLMv3

cd layoutlmv3_ner
python train.py         # Train with labeled data
python predict.py       # Inference on new document

Donut

cd donut_inference
python infer.py --image sample_invoice.jpg


⸻

📦 Sample Dataset

We include small samples of:
	•	Invoices in English and Chinese
	•	Form-like PDFs
	•	OCR annotations in words, bboxes, and labels format

You can annotate your own documents using Label Studio or Doccano.

⸻

📊 Model Comparison Table

| 对比维度           | PaddleOCR         | LayoutLMv3              | Donut                  |
|--------------------|-------------------|--------------------------|------------------------|
| **输入**           | 图像              | OCR后的文字 + bbox      | 图像                   |
| **输出**           | 文本 + 坐标       | 实体标签（BIO）         | JSON结构               |
| **是否依赖OCR**    | ✅ 内置OCR        | ✅ 依赖外部OCR           | ❌ 无需OCR             |
| **中文支持**       | ✅ 极好           | ✅ 依赖中文OCR           | ⚠️ 微调后可用         |
| **训练需求**       | ❌ 无需训练       | ✅ BIO标签数据           | ✅ JSON标签数据        |
| **适用文档格式**   | 任意              | 半结构化                 | 模板固定文档           |
| **推理速度**       | ✅ 快             | 中等                     | 慢                     |
| **易用性**         | 极高              | 中等                     | 中等偏高               |


⸻

📖 References
	•	PaddleOCR GitHub
	•	LayoutLMv3 Paper (Microsoft)
	•	Donut Paper (NAVER)
	•	HuggingFace Transformers
⸻

👩‍🏫 Instructor / Maintainer

This course is maintained by May (Xiaoting Zhou),
AI researcher, document intelligence developer, and educator.
	•	💌 Contact: may200852@gmail.com
	•	📬 Issues or suggestions? Open a GitHub Issue

⸻

🔖 License

MIT License. Free to use for teaching, research, or prototyping purposes.

---
