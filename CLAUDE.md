# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is an AI OCR (Optical Character Recognition) research repository focused on Chinese invoice processing using multiple deep learning models:

### Main Model Architectures
- **LayoutLMv3**: Primary NER model for structured document understanding (`layoutlmv3_ner/`)
- **Donut**: Vision-based document AI model for invoice extraction (`donut_invoice_model/`)  
- **LayoutXLM**: Multilingual layout model (`layoutxlm-base/`)

### Core Components Structure
- **Data Processing Pipeline**: `layoutlmv3_ner/Lec_5/src/data_processor.py` - Handles PDF-to-image conversion, OCR extraction using EasyOCR, and annotation creation
- **Training Framework**: `layoutlmv3_ner/Lec_5/src/layoutlm_trainer.py` - LayoutLMv3 token classification training
- **Inference Pipeline**: `layoutlmv3_ner/Lec_5/src/inference_pipeline.py` - Production inference for trained models
- **Main Tutorial**: `layoutlmv3_ner/Lec_5/tutorial_demo.py` - Complete end-to-end demonstration

## Development Commands

### Environment Setup
```bash
# Install LayoutLMv3 dependencies
pip install -r layoutlmv3_ner/Lec_5/requirements.txt

# Install Donut dependencies  
pip install -r donut_invoice_model/Lec_4/requirements.txt
```

### Training Commands
```bash
# LayoutLMv3 training (main tutorial)
cd layoutlmv3_ner/Lec_5
python tutorial_demo.py

# Donut model training
cd donut_invoice_model/Lec_4
python train.py

# Custom LayoutLMv3 training
cd layoutlmv3_ner/Lec_5/src
python layoutlm_trainer.py
```

### Data Processing
```bash
# Process new invoice data
cd layoutlmv3_ner/Lec_5
python src/data_processor.py

# Fix bbox coordinates (utility)
python fix_bbox_coordinates.py
```

### Inference
```bash
# Run inference on new documents
cd layoutlmv3_ner/Lec_5
python src/inference_pipeline.py

# Donut prediction
cd donut_invoice_model/Lec_4
python predict.py
```

## Data Architecture

### Directory Structure
- `data/raw/` - Original PDF invoices (測試股份有限公司_*.pdf)
- `data/images/` - Converted images from PDFs  
- `data/training/` - Training annotations (label_*.json files)
- `data/validation/` - Validation annotations
- `data/annotation/` - Generated annotation files with timestamps

### Label Schema
The system uses BIO tagging for Named Entity Recognition:
- `B-InvoiceNo`, `I-InvoiceNo` - Invoice number
- `B-InvoiceDate`, `I-InvoiceDate` - Invoice date
- `B-Currency`, `I-Currency` - Currency type
- `B-AmountwithTax`, `I-AmountwithTax` - Tax-inclusive amount
- `B-AmountwithoutTax`, `I-AmountwithoutTax` - Tax-exclusive amount  
- `B-Tax`, `I-Tax` - Tax amount
- `O` - Outside entity

## Key Technical Details

### OCR Engine
- **Primary**: EasyOCR with Chinese Simplified and English support
- **Alternative**: PaddleOCR (legacy, still in some modules)
- **Fallback**: Tesseract for specific use cases

### Model Checkpoints
- Models saved to `models/tutorial_invoice_layoutlmv3/` during training
- Final models in `models/final/` directory
- Donut models in `donut_invoice_model/Lec_4/donut-invoice-finetuned/`

### Training Configuration
- LayoutLMv3: 2-3 epochs for demo, max_length=512, supports GPU/CPU
- Donut: 10 epochs, batch_size=2, learning_rate=5e-5
- Early stopping enabled for both models

### Performance Monitoring
- Training logs saved with timestamps
- Built-in performance diagnostics in `src/performance_diagnostics.py`
- Monitoring capabilities in `monitoring.py`

## Important File Paths

### Current Working Data
- Training data: `layoutlmv3_ner/Lec_5/data/training/demo_train.json`
- Main config: Hard-coded paths in `tutorial_demo.py:33`
- Invoice PDFs: `layoutlmv3_ner/Lec_5/data/raw/測試股份有限公司_[1-11].pdf`

### Processing Pipeline
1. PDF → Images: `data_processor.pdf_to_images()`
2. Images → OCR: `data_processor.extract_text_with_positions()`  
3. OCR + Labels → Annotations: `data_processor.create_training_annotations()`
4. Annotations → Model Training: `layoutlm_trainer.py`
5. Trained Model → Inference: `inference_pipeline.py`

## Development Notes

- All file paths are currently absolute and Mac-specific (`/Users/xiaotingzhou/...`)
- The system processes 11 sample invoices (測試股份有限公司_1.pdf through _11.pdf)
- Annotation files are automatically generated with timestamps
- GPU support available but CPU fallback implemented
- Multiprocessing configured with 'spawn' method to prevent semaphore leaks