#!/usr/bin/env python3
"""
Simple test of OCR extraction with proper bbox normalization
"""
import json
import logging

class SimpleOCRTest:
    def __init__(self):
        # Mock EasyOCR results for testing
        self.mock_results = [
            # Format: (bbox_points, text, confidence)
            ([[178, 124], [254, 124], [254, 139], [178, 139]], "TE ST", 0.51),
            ([[390, 125], [503, 125], [503, 140], [390, 140]], "Test Co.,Ltd.", 0.55),
            ([[390, 159], [457, 159], [457, 173], [390, 173]], "Invoice", 0.98),
        ]
        
        # Mock image dimensions (from log: 2481x3508)
        self.img_width = 2481
        self.img_height = 3508
        
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
    
    def normalize_bbox_coordinates(self):
        """Test bbox normalization"""
        print(f"🖼️ 图像尺寸: {self.img_width} x {self.img_height}")
        print(f"📋 测试 {len(self.mock_results)} 个OCR结果的坐标归一化:\n")
        
        normalized_results = []
        
        for i, (bbox_points, text, confidence) in enumerate(self.mock_results):
            print(f"文本 {i+1}: '{text}'")
            print(f"  原始bbox点: {bbox_points}")
            
            # Extract coordinates like the real code does
            x_coords = [point[0] for point in bbox_points]
            y_coords = [point[1] for point in bbox_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            print(f"  原始坐标范围: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
            
            # Apply normalization (same as fixed code)
            norm_x1 = max(0, min(1000, int((min_x / self.img_width) * 1000)))
            norm_y1 = max(0, min(1000, int((min_y / self.img_height) * 1000)))
            norm_x2 = max(0, min(1000, int((max_x / self.img_width) * 1000)))
            norm_y2 = max(0, min(1000, int((max_y / self.img_height) * 1000)))
            
            # Ensure x2 > x1, y2 > y1
            if norm_x2 <= norm_x1:
                norm_x2 = min(1000, norm_x1 + 1)
            if norm_y2 <= norm_y1:
                norm_y2 = min(1000, norm_y1 + 1)
            
            normalized_bbox = [norm_x1, norm_y1, norm_x2, norm_y2]
            
            print(f"  归一化坐标: {normalized_bbox}")
            
            # Validate
            if all(isinstance(coord, int) and 0 <= coord <= 1000 for coord in normalized_bbox):
                print(f"  ✅ 坐标有效")
                normalized_results.append({
                    'text': text,
                    'bbox': normalized_bbox,
                    'confidence': confidence
                })
            else:
                print(f"  ❌ 坐标无效")
            
            print("")
        
        print(f"📊 测试结果:")
        print(f"  成功归一化: {len(normalized_results)}/{len(self.mock_results)}")
        
        # Save test results
        with open('test_normalized.json', 'w', encoding='utf-8') as f:
            json.dump(normalized_results, f, ensure_ascii=False, indent=2)
        
        print(f"  结果已保存到: test_normalized.json")
        
        return len(normalized_results) == len(self.mock_results)

if __name__ == '__main__':
    tester = SimpleOCRTest()
    success = tester.normalize_bbox_coordinates()
    
    if success:
        print("\n🎉 归一化测试成功！")
    else:
        print("\n❌ 归一化测试失败！")