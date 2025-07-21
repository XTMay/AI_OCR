import re
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def clean_json_string(json_str: str) -> str:
    """清理生成的 JSON 字符串"""
    # 移除多余的空白字符
    json_str = re.sub(r'\s+', ' ', json_str.strip())
    
    # 修复常见的 JSON 格式问题
    json_str = re.sub(r',\s*}', '}', json_str)  # 移除尾随逗号
    json_str = re.sub(r',\s*]', ']', json_str)  # 移除数组尾随逗号
    
    # 确保字符串被正确引用
    json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    
    return json_str

def infer_currency(data: Dict[str, Any]) -> str:
    """从数据中推断货币类型"""
    # 检查现有的 Currency 字段
    if data.get("Currency"):
        return data["Currency"]
    
    # 从文本内容中推断
    text_content = json.dumps(data, ensure_ascii=False).upper()
    
    currency_patterns = {
        "USD": [r"USD", r"\$", r"DOLLAR", r"美元"],
        "CNY": [r"CNY", r"RMB", r"¥", r"人民币", r"元"],
        "EUR": [r"EUR", r"€", r"EURO"],
        "GBP": [r"GBP", r"£", r"POUND"]
    }
    
    for currency, patterns in currency_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_content):
                return currency
    
    return "USD"  # 默认货币

def infer_tax(data: Dict[str, Any]) -> float:
    """推断税金"""
    # 如果已有税金信息，直接返回
    if "Tax" in data and data["Tax"] is not None:
        try:
            return float(data["Tax"])
        except (ValueError, TypeError):
            pass
    
    # 尝试从 TotalAmount 和 NetAmount 计算
    total_amount = data.get("TotalAmount")
    net_amount = data.get("NetAmount")
    
    if total_amount is not None and net_amount is not None:
        try:
            tax = float(total_amount) - float(net_amount)
            return round(tax, 2)
        except (ValueError, TypeError):
            pass
    
    # 尝试从 Items 计算总税金
    items = data.get("Items", [])
    if items:
        total_tax = 0
        for item in items:
            item_tax = item.get("Tax", 0)
            if item_tax:
                try:
                    total_tax += float(item_tax)
                except (ValueError, TypeError):
                    pass
        if total_tax > 0:
            return round(total_tax, 2)
    
    # 默认税金为 0
    return 0.0

def infer_payment_term(data: Dict[str, Any]) -> str:
    """推断付款条件"""
    if data.get("PaymentTerm"):
        return data["PaymentTerm"]
    
    # 从文本中查找付款条件模式
    text_content = json.dumps(data, ensure_ascii=False).upper()
    
    payment_patterns = {
        r'(\d+)\s*DAYS?': lambda m: f"{m.group(1)}DAYS",
        r'NET\s*(\d+)': lambda m: f"NET{m.group(1)}",
        r'(\d+)\s*天': lambda m: f"{m.group(1)}DAYS",
        r'现金': lambda m: "CASH",
        r'CASH': lambda m: "CASH"
    }
    
    for pattern, formatter in payment_patterns.items():
        match = re.search(pattern, text_content)
        if match:
            return formatter(match)
    
    return "NET30"  # 默认付款条件

def normalize_amounts(data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化金额字段"""
    amount_fields = ["TotalAmount", "NetAmount", "Tax", "TotalQty"]
    
    for field in amount_fields:
        if field in data and data[field] is not None:
            try:
                # 尝试转换为数字
                value = str(data[field]).replace(",", "").replace(" ", "")
                if field == "TotalQty":
                    data[field] = int(float(value))
                else:
                    data[field] = round(float(value), 2)
            except (ValueError, TypeError):
                logger.warning(f"无法转换字段 {field} 的值: {data[field]}")
    
    # 处理 Items 中的金额
    if "Items" in data and isinstance(data["Items"], list):
        for item in data["Items"]:
            item_amount_fields = ["Qty", "UnitPrice", "Amount", "Tax"]
            for field in item_amount_fields:
                if field in item and item[field] is not None:
                    try:
                        value = str(item[field]).replace(",", "").replace(" ", "")
                        if field == "Qty":
                            item[field] = int(float(value))
                        else:
                            item[field] = round(float(value), 2)
                    except (ValueError, TypeError):
                        logger.warning(f"无法转换项目字段 {field} 的值: {item[field]}")
    
    return data

def validate_invoice_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """验证和修复发票数据"""
    # 确保必需字段存在
    required_fields = {
        "InvoiceNo": "",
        "InvoiceDate": "",
        "Currency": "USD",
        "Items": [],
        "TotalAmount": 0.0,
        "Tax": 0.0
    }
    
    for field, default_value in required_fields.items():
        if field not in data or data[field] is None:
            data[field] = default_value
    
    # 验证日期格式
    if data.get("InvoiceDate"):
        date_str = str(data["InvoiceDate"])
        # 尝试标准化日期格式
        date_patterns = [
            (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', r'\1-\2-\3'),
            (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', r'\3-\1-\2')
        ]
        
        for pattern, replacement in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                data["InvoiceDate"] = re.sub(pattern, replacement, date_str)
                break
    
    return data

def infer_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """主要的字段推理函数"""
    if not isinstance(data, dict):
        return data
    
    try:
        # 1. 标准化金额
        data = normalize_amounts(data)
        
        # 2. 推断货币
        data["Currency"] = infer_currency(data)
        
        # 3. 推断税金
        data["Tax"] = infer_tax(data)
        
        # 4. 推断付款条件
        if not data.get("PaymentTerm"):
            data["PaymentTerm"] = infer_payment_term(data)
        
        # 5. 验证和修复数据
        data = validate_invoice_data(data)
        
        # 6. 计算总数量（如果缺失）
        if "Items" in data and isinstance(data["Items"], list) and not data.get("TotalQty"):
            total_qty = sum(item.get("Qty", 0) for item in data["Items"] if isinstance(item.get("Qty"), (int, float)))
            data["TotalQty"] = int(total_qty)
        
        logger.info("字段推理完成")
        return data
        
    except Exception as e:
        logger.error(f"字段推理过程中出错: {e}")
        return data

def evaluate_prediction(pred: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """评估预测结果"""
    metrics = {
        "exact_match": 0.0,
        "field_accuracy": 0.0,
        "amount_accuracy": 0.0
    }
    
    # 完全匹配
    if pred == ground_truth:
        metrics["exact_match"] = 1.0
    
    # 字段级别准确率
    total_fields = len(ground_truth)
    correct_fields = 0
    
    for key, gt_value in ground_truth.items():
        if key in pred and pred[key] == gt_value:
            correct_fields += 1
    
    metrics["field_accuracy"] = correct_fields / total_fields if total_fields > 0 else 0.0
    
    # 金额字段准确率
    amount_fields = ["TotalAmount", "Tax", "TotalQty"]
    total_amount_fields = sum(1 for field in amount_fields if field in ground_truth)
    correct_amount_fields = 0
    
    for field in amount_fields:
        if field in ground_truth and field in pred:
            try:
                if abs(float(pred[field]) - float(ground_truth[field])) < 0.01:
                    correct_amount_fields += 1
            except (ValueError, TypeError):
                pass
    
    metrics["amount_accuracy"] = correct_amount_fields / total_amount_fields if total_amount_fields > 0 else 0.0
    # % 置信度 > 50% --> output
    # % 置信度 < 50% --> RF

    return metrics