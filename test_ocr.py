import easyocr
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)

# 初始化OCR
ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# 测试图像
image_path = "/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images/測試股份有限公司_1_page_1.jpg"

try:
    results = ocr.readtext(image_path)
    print(f"检测到 {len(results)} 个文本区域:")
    
    for i, (bbox, text, confidence) in enumerate(results):
        print(f"文本 {i}: '{text}'")
        print(f"  置信度: {confidence}")
        print(f"  原始bbox: {bbox}")
        
        # 转换bbox
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        normalized_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        print(f"  标准化bbox: {normalized_bbox}")
        print()
        
except Exception as e:
    print(f"OCR失败: {e}")
    import traceback
    traceback.print_exc()