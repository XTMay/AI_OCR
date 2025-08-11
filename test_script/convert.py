# 新版 執行整批
import json

def tokenize(text, label):
    # TODO 标签名要与Json中的标签名一致
    chinese_labels = {"BuyerName", "VendorName", "InvoiceDate", "Currency", "AmountWithTax", "AmountWithoutTax", "Tax"}
    return list(text) if label in chinese_labels else text.split()

def process_annotation(annotation):
    results = annotation["result"]
    trans_map, bbox_map, label_map = {}, {}, {}
    used_ids, relation_pairs = set(), []

    for item in results:
        if item.get("type") == "relation":
            relation_pairs.append((item["from_id"], item["to_id"]))
            continue

        item_id = str(item.get("id"))
        from_name = item.get("from_name")
        value = item.get("value")

        if from_name == "transcription":
            trans_map[item_id] = value["text"][0]
        elif from_name == "bbox":
            bbox_map[item_id] = {
                "x": value["x"],
                "y": value["y"],
                "width": value["width"],
                "height": value["height"],
                "original_width": item["original_width"],
                "original_height": item["original_height"]
            }
        elif from_name == "label":
            label_map[item_id] = value["rectanglelabels"][0]

    # 處理 relation，保留每個 bbox 並依順序標註 BIO
    merged = []
    for from_id, to_id in relation_pairs:
        used_ids.update([from_id, to_id])
        for idx, item_id in enumerate([from_id, to_id]):
            text = trans_map.get(item_id, "")
            label = label_map.get(from_id, "O")  # 使用 relation 的 from_id label
            b = bbox_map[item_id]
            x0 = b["x"] / 100 * b["original_width"]
            y0 = b["y"] / 100 * b["original_height"]
            x1 = (b["x"] + b["width"]) / 100 * b["original_width"]
            y1 = (b["y"] + b["height"]) / 100 * b["original_height"]
            bbox = [int(x0), int(y0), int(x1), int(y1)]
            merged.append((text, bbox, label, idx == 0))  # 是否為第一個 token

    # 處理非 relation 的標註
    for item_id in trans_map:
        if item_id in used_ids:
            continue
        text = trans_map[item_id]
        label = label_map.get(item_id, "O")
        b = bbox_map[item_id]
        x0 = b["x"] / 100 * b["original_width"]
        y0 = b["y"] / 100 * b["original_height"]
        x1 = (b["x"] + b["width"]) / 100 * b["original_width"]
        y1 = (b["y"] + b["height"]) / 100 * b["original_height"]
        bbox = [int(x0), int(y0), int(x1), int(y1)]
        merged.append((text, bbox, label, True))  # 單獨標註視為 B

    # BIO 標註
    tokens, bboxes, ner_tags = [], [], []
    for text, bbox, label, is_first in merged:
        words = tokenize(text, label)
        for i, word in enumerate(words):
            tokens.append(word)
            bboxes.append(bbox)
            if label == "O":
                ner_tags.append("O")
            else:
                ner_tags.append(f"{'B' if is_first and i == 0 else 'I'}-{label}")

    return {
        "id": annotation.get("id"),
        "tokens": tokens,
        "bboxes": bboxes,
        "ner_tags": ner_tags
    }


# 載入 Label Studio 匯出的 JSON
with open("/Users/xiaotingzhou/Documents/Lectures/AI_OCR/test_script/測試股份有限公司_標註後輸出結果.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 處理所有任務
bio_data = []
for task in data:
    for annotation in task.get("annotations", []):
        bio_data.append(process_annotation(annotation))

# 儲存為 LayoutLMv3 BIO 格式
with open("測試股份有限公司_轉換為BIO結果.json", "w", encoding="utf-8") as f:
    json.dump(bio_data, f, ensure_ascii=False, indent=2)

print("已成功整合所有標註資料並輸出為 layoutlm_bio_final.json")

#   1. 中文分词问题：对中文文本按字符分割可能不合适
#   2. 缺少错误处理：没有文件存在性检查
#   3. 输出消息错误：提到错误的文件名
#   4. BIO标记逻辑：relation处理可能导致标记错误

#   ✅ 建议修改：输出格式
#   [
#     {
#       "image_path": "图片路径",
#       "entities": [
#         {
#           "text": "发票号码文本",
#           "bbox": [x0, y0, x1, y1],
#           "label": "B-InvoiceNo"
#         }
#       ]
#     }
#   ]

#   而不是当前的token级别格式。建议重写这个转换脚本以匹配LayoutLMv3的训练数据格式要求。