import json
import os
from PIL import Image
from pathlib import Path

def normalize_bbox_coordinates(annotation_files, image_dir):
    """Normalize bbox coordinates to 0-1000 range"""
    
    for file_path in annotation_files:
        print(f"Processing: {file_path}")
        
        # Load annotations
        with open(file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Process each annotation
        for ann in annotations:
            image_path = ann['image_path']
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                
                # Normalize bbox coordinates
                bbox = ann['bbox']
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    
                    # Normalize to 0-1000 range
                    normalized_bbox = [
                        int((x1 / img_width) * 1000),
                        int((y1 / img_height) * 1000),
                        int((x2 / img_width) * 1000),
                        int((y2 / img_height) * 1000)
                    ]
                    
                    # Ensure coordinates are within 0-1000 range
                    normalized_bbox = [max(0, min(1000, coord)) for coord in normalized_bbox]
                    
                    # Update annotation
                    ann['bbox'] = normalized_bbox
                    
                    print(f"  Original: {bbox} -> Normalized: {normalized_bbox}")
                    
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue
        
        # Save normalized annotations
        backup_path = file_path.replace('.json', '_backup.json')
        os.rename(file_path, backup_path)
        print(f"  Backup saved: {backup_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        print(f"  Updated: {file_path}")

if __name__ == '__main__':
    # List of annotation files to fix
    annotation_files = [
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/all_annotations_20250724_191140.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/all_annotations_20250724_194408.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/B-AmountwithTax_annotations_20250724_191140.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/B-AmountwithTax_annotations_20250724_194408.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/B-Currency_annotations_20250724_191140.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/B-Currency_annotations_20250724_194408.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/B-InvoiceNo_annotations_20250724_191140.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/B-InvoiceNo_annotations_20250724_194408.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/O_annotations_20250724_191140.json',
        '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/annotation/O_annotations_20250724_194408.json'
    ]
    
    image_dir = '/Users/xiaotingzhou/Documents/Lectures/AI_OCR/layoutlmv3_ner/Lec_5/data/images'
    
    normalize_bbox_coordinates(annotation_files, image_dir)
    print("\n✅ All annotation files have been normalized!")