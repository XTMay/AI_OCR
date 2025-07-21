# Donut 多语言发票信息抽取系统

基于 Donut (Document Understanding Transformer) 的多语言发票信息抽取系统，支持中英文混排发票的结构化信息提取。

## 🚀 功能特点

- **多语言支持**: 支持中英文混排发票
- **结构化输出**: 输出标准 JSON 格式的发票信息
- **智能推理**: 自动推断税金、货币类型等隐含字段
- **灵活配置**: 支持自定义训练参数和模型配置
- **批量处理**: 支持单张和批量发票处理

## 📋 支持的字段

```json
{
  "InvoiceNo": "发票号码",
  "InvoiceDate": "发票日期",
  "Currency": "货币类型",
  "Items": [
    {
      "Description": "商品描述",
      "P.O.NO": "采购订单号",
      "Qty": "数量",
      "UnitPrice": "单价",
      "Amount": "金额",
      "DeliveryNoteNo": "送货单号"
    }
  ],
  "TotalQty": "总数量",
  "TotalAmount": "总金额",
  "Tax": "税金",
  "PaymentTerm": "付款条件",
  "Beneficiary": {
    "Name": "受益人名称",
    "Bank": "银行名称",
    "BankAddress": "银行地址",
    "BankAccount": "银行账号",
    "ContactPerson": "联系人"
  }
}
```

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

## 📁 数据准备

将训练数据按以下结构组织：
