import os
import json
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import re
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class TesseractInvoiceExtractor:
    def __init__(self):
        """初始化发票提取器"""
        try:
            # 测试pytesseract是否可用
            pytesseract.get_tesseract_version()
            print("Tesseract初始化成功")
        except Exception as e:
            print(f"Tesseract初始化失败: {e}")
            print("请安装tesseract: brew install tesseract tesseract-lang")
    
    def pdf_to_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """将PDF转换为图片"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), 'converted_images')
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 转换PDF为图片
            images = convert_from_path(pdf_path, dpi=300)  # 提高DPI
            image_paths = []
            
            for i, image in enumerate(images):
                image_path = os.path.join(output_dir, f'invoice_page_{i+1}.jpg')
                image.save(image_path, 'JPEG', quality=95)
                image_paths.append(image_path)
                print(f"已保存页面 {i+1}: {image_path}")
            
            return image_paths
        except Exception as e:
            print(f"PDF转换失败: {e}")
            return []
    
    # 在extract_text_with_tesseract方法中添加更多预处理步骤
    def extract_text_with_tesseract(self, image_path: str) -> List[str]:
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # CLAHE (限制对比度自适应直方图均衡化)：解决光照不均问题
            # 增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            # 形态学操作：去除小噪点，连接断裂文字
            # 形态学操作去除噪声
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # OTSU二值化：自动选择最佳阈值
            # 二值化
            _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 使用不同的PSM模式
            configs = [
                r'--oem 3 --psm 6 -l chi_sim+eng',
                r'--oem 3 --psm 4 -l chi_sim+eng',
                r'--oem 3 --psm 3 -l chi_sim+eng'
            ]
            
            all_text = []
            for config in configs:
                text = pytesseract.image_to_string(binary, config=config)
                all_text.extend([line.strip() for line in text.split('\n') if line.strip()])
            
            return list(set(all_text))  # 去重
            
        except Exception as e:
            print(f"OCR处理失败: {e}")
            return []
            
    def extract_invoice_fields(self, lines: List[str]) -> Dict[str, str]:
        """从识别的文本中提取发票字段"""
        # 将所有文本合并
        full_text = ' '.join(lines)
        print(f"识别的完整文本: {full_text[:200]}...")  # 显示前200个字符
        
        # 初始化结果
        result = {
            "InvoiceNo": "",
            "InvoiceDate": "",
            "Currency": "CNY",
            "Amount with Tax": "",
            "Amount without Tax": "",
            "Tax": ""
        }
        
        # 发票号码模式
        invoice_patterns = [
            r'([A-Z]\s*\d{12})',
            r'发票号码[：:]*\s*([A-Z]?\d+)',
            r'Invoice\s*No[.：:]*\s*([A-Z]?\d+)',
            r'([A-Z]\d{8,})',
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                result["InvoiceNo"] = match.group(1).replace(' ', '')
                break
        
        # 日期模式
        date_patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}/\d{1,2}/\d{1,2})',
            r'(\d{4}-\d{1,2}-\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, full_text)
            if match:
                result["InvoiceDate"] = match.group(1)
                break
        
        # 金额提取
        amount_patterns = [
            r'合计[：:]*\s*[￥¥]?([\d,]+\.?\d*)',
            r'总计[：:]*\s*[￥¥]?([\d,]+\.?\d*)',
            r'价税合计[：:]*\s*[￥¥]?([\d,]+\.?\d*)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, full_text)
            if match:
                result["Amount with Tax"] = match.group(1).replace(',', '')
                break
        
        # 如果没有找到特定标签的金额，查找数字
        if not result["Amount with Tax"]:
            numbers = re.findall(r'\b\d{2,8}(?:\.\d{1,2})?\b', full_text)
            if numbers:
                amounts = [float(n) for n in numbers if float(n) > 10]
                if amounts:
                    max_amount = max(amounts)
                    result["Amount with Tax"] = str(int(max_amount) if max_amount.is_integer() else max_amount)
        
        # 税额计算
        if result["Amount with Tax"] and not result["Tax"]:
            try:
                with_tax = float(result["Amount with Tax"])
                without_tax = with_tax / 1.13  # 假设13%税率
                tax = with_tax - without_tax
                result["Amount without Tax"] = str(round(without_tax, 2))
                result["Tax"] = str(round(tax, 2))
            except ValueError:
                result["Amount without Tax"] = result["Amount with Tax"]
                result["Tax"] = "0"
        
        return result
    
    def extract_invoice_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """从PDF提取发票信息的主函数"""
        print(f"开始处理PDF文件: {pdf_path}")
        
        # 1. 转换PDF为图片
        image_paths = self.pdf_to_images(pdf_path)
        
        if not image_paths:
            return {"error": "PDF转换失败"}
        
        # 2. 处理第一页
        image_path = image_paths[0]
        print(f"处理图片: {image_path}")
        
        try:
            # 3. 使用Tesseract提取文本
            lines = self.extract_text_with_tesseract(image_path)
            
            if not lines:
                return {"error": "OCR识别失败"}
            
            # 4. 提取发票字段
            result = self.extract_invoice_fields(lines)
            
            print("提取结果:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            return result
            
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            return {"error": str(e)}

def main():
    # PDF文件路径
    pdf_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/data/測試股份有限公司.pdf"
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return
    
    # 创建提取器
    extractor = TesseractInvoiceExtractor()
    
    # 提取发票信息
    result = extractor.extract_invoice_from_pdf(pdf_path)
    
    # 保存结果
    output_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/tesseract_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    print("\n最终JSON结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()