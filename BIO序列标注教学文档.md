# 发票OCR文本BIO序列标注教学文档

## 摘要

本文档为发票OCR文本BIO序列标注的完整教学指南，面向标注团队提供详细的操作规范和示例。文档基于提供的示例文件（包含发票号码Y 309824263008、日期2025年6月30日等字段）生成多种标注场景，覆盖正常情况、OCR错误、跨行切分、手写识别等复杂情况。标注员可直接使用本文档进行实际标注工作，包含完整的质量保证流程和工具配置指南。

---

# 1. 引言与任务定义

## 1.1 任务目标

本项目旨在对发票OCR输出文本进行BIO（Begin-Inside-Outside）序列标注，用于训练多模态Transformer模型（结合OCR文本、边界框坐标和图像信息）。我们需要从发票中准确抽取以下关键信息字段：

## 1.2 字段名与统一Label名称

| 字段含义 | 统一标签名称 | 示例值 |
|---------|-------------|--------|
| 发票号码 | INVOICE_NO | Y 309824263008 |
| 开票日期 | DATE | 2025年6月30日 |
| 币种 | CURRENCY | USD |
| 含税金额 | AMOUNT_WITH_TAX | 300 |
| 不含税金额 | AMOUNT_WITHOUT_TAX | 300 |
| 税额 | TAX | 0 |

## 1.3 BIO标签格式

- **B-[LABEL]**: 实体的开始标记，如 `B-INVOICE_NO`
- **I-[LABEL]**: 实体的内部标记，如 `I-INVOICE_NO`
- **O**: 非实体标记（Outside）

**示例：**
```
文本: 发票号码: Y 309824263008
标签: O O B-INVOICE_NO I-INVOICE_NO
```

---

# 2. 标注总原则

## 2.1 核心原则一：基于OCR输出标注

**重要：标注必须基于OCR实际输出，不要在标注阶段人为修正为"理想正确文本"。**

- ✅ 正确做法：OCR输出 "Y 3O9824263OO8"，仍标注为实体
- ❌ 错误做法：标注时改成 "Y 309824263008"

后处理阶段再进行文本纠正和归一化。

## 2.2 核心原则二：跨Token连续实体标注

若实体被切分为多个token（包括跨行或跨边界框），仍按连续实体进行BIO标注。

**示例：**
```
OCR输出跨行：
第1行: Y 309824
第2行: 263008

标注：
Y: B-INVOICE_NO
309824: I-INVOICE_NO  
263008: I-INVOICE_NO
```

## 2.3 核心原则三：完全不可识别内容

完全不可识别的乱码（如###、完全空白、乱码符号）标为O，并添加标注器注释`__REVIEW__`。

**示例：**
```
OCR: 发票号码: ### @#$%
标注: O O O O
注释: __REVIEW__ 发票号码区域OCR失败
```

## 2.4 核心原则四：部分可识别内容

OCR输出部分可识别（如12I4O678），仍标注为实体，不在标注阶段纠错。

## 2.5 标注员快速决策树

遇到token时按以下顺序判断：

1. **是否有前置关键词？**（发票号/Invoice No/开票日期/Total等）→ 是：检查后续token
2. **是否在表头或汇总区域？** → 是：重点关注
3. **是否为数字且带货币符号？** → 是：可能为金额
4. **是否为日期格式？** → 是：标注为DATE
5. **是否为币种标识？** → 是：标注为CURRENCY
6. **无法判断** → 标注为O，添加__REVIEW__标记

---

# 3. Tokenization与粒度建议

## 3.1 中文发票Tokenization

**建议：字符级(character-level)或细粒度分词**

中文发票示例：
```
原文: 2025年06月30日
字符级: ['2','0','2','5','年','0','6','月','3','0','日']
BIO:   [B-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE]

原文: 金额：300.00
字符级: ['金','额','：','3','0','0','.','0','0']  
BIO:   [O,O,O,B-AMOUNT_WITH_TAX,I-AMOUNT_WITH_TAX,I-AMOUNT_WITH_TAX,I-AMOUNT_WITH_TAX,I-AMOUNT_WITH_TAX,I-AMOUNT_WITH_TAX]
```

## 3.2 英文发票Tokenization

**建议：空格/标点分割为主**

英文发票示例：
```
原文: Invoice No: Y 309824263008
空格分割: ['Invoice', 'No:', 'Y', '309824263008']
BIO:     [O, O, B-INVOICE_NO, I-INVOICE_NO]
```

## 3.3 两套标准Tokenization示例

### 方案1：Whitespace Tokenization（英文优先）
```python
# 伪代码
def whitespace_tokenize(text):
    return text.split()

示例: "Total Amount: 300 USD"
tokens: ['Total', 'Amount:', '300', 'USD']
```

### 方案2：Character/CJK-friendly Tokenization（中文优先）
```python
# 伪代码  
def cjk_tokenize(text):
    tokens = []
    for char in text:
        if char.isalnum() or char in '年月日￥$':
            tokens.append(char)
        elif char.isspace():
            continue
        else:
            tokens.append(char)
    return tokens

示例: "金额：300元"
tokens: ['金','额','：','3','0','0','元']
```

## 3.4 标注工具设置建议

- **doccano**: 选择"字符选择"模式，支持任意粒度标注
- **Label Studio**: 配置自定义tokenizer或使用字符级选择
- 如工具不支持细粒度，建议导出后用脚本重新对齐

---

# 4. 每个字段的标注规则与归一化

## 4.1 INVOICE_NO（发票号码）

### 定义
包含字母、数字、短横线、空格的发票唯一标识符，通常出现在"发票号"、"Invoice No"、"发票号码"、"Invoice #"等关键词附近。

### 常见OCR错误
- 数字混淆：0↔O, 1↔I, 5↔S, 6↔G
- 空格误插入或丢失
- 字符断开跨行

### 标注示例

#### 示例1：正常情况
```
OCR原始：发票号码: Y 309824263008
Tokens:  ['发票号码', ':', 'Y', '309824263008']
BIO:     [O, O, B-INVOICE_NO, I-INVOICE_NO]
```

#### 示例2：OCR错误
```
OCR错误：发票号码: Y 3O98242 63OO8
Tokens:  ['发票号码', ':', 'Y', '3O98242', '63OO8'] 
BIO:     [O, O, B-INVOICE_NO, I-INVOICE_NO, I-INVOICE_NO]
标注依据：虽有OCR错误(O/0混淆，空格错插)，仍标注为实体
```

#### 示例3：跨行情况
```
第1行：Y 3098242
第2行：63008
Tokens: ['Y', '3098242', '63008']
BIO:    [B-INVOICE_NO, I-INVOICE_NO, I-INVOICE_NO]
标注依据：跨行但语义连续，保持B-I序列
```

### 后处理归一化代码
```python
def normalize_invoice_no(raw_text):
    # 全角转半角
    text = raw_text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    # OCR常见混淆修正(启发式)
    text = text.replace('O','0').replace('I','1').replace('S','5')
    # 移除非字母数字字符
    text = re.sub(r'[^A-Za-z0-9\-]', '', text)
    return text

# 验证正则
invoice_pattern = r'^[A-Za-z0-9\-]{6,30}$'
```

## 4.2 DATE（开票日期）

### 定义
表示发票开具日期，支持多种格式。

### 常见格式
- 中文：2025年6月30日、2025年06月30日
- 数字：2025/06/30、2025-06-30、30.06.2025
- 英文：Jun 30, 2025、June 30, 2025

### OCR错误
- 年月日文字与数字粘连或分离
- 数字识别错误
- 标点符号混淆

### 标注示例

#### 示例1：标准中文格式
```
OCR原始：开票日期：2025年6月30日
Tokens:  ['开票日期', '：', '2025年6月30日']
BIO:     [O, O, B-DATE]

字符级tokenization:
Tokens:  ['开','票','日','期','：','2','0','2','5','年','6','月','3','0','日']
BIO:     [O,O,O,O,O,B-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE,I-DATE]
```

#### 示例2：OCR分割错误
```
OCR错误：2025 年6月 30日
Tokens:  ['2025', '年6月', '30日']
BIO:     [B-DATE, I-DATE, I-DATE]
标注依据：虽被分割，仍为连续日期实体
```

### 后处理代码
```python
import re
from dateutil import parser

def normalize_date(raw_text):
    # 中文日期处理
    chinese_date = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', raw_text)
    if chinese_date:
        year, month, day = chinese_date.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # 其他格式尝试解析
    try:
        parsed = parser.parse(raw_text, fuzzy=True)
        return parsed.strftime('%Y-%m-%d')
    except:
        return None  # 需人工审查
```

## 4.3 CURRENCY（币种）

### 定义
货币种类标识符，包括符号和代码形式。

### 支持格式
- 代码：USD、CNY、EUR、JPY、GBP等
- 符号：$、￥、€、£等
- 中文：美元、人民币、欧元等

### OCR错误
- USD → US D、U$D、US0
- $ → S、5
- ￥ → Y、¥

### 标注示例

#### 示例1：标准代码
```
OCR原始：Currency: USD
Tokens:  ['Currency:', 'USD']
BIO:     [O, B-CURRENCY]
```

#### 示例2：货币符号
```
OCR原始：Total: $300
Tokens:  ['Total:', '$300']
BIO:     [O, B-AMOUNT_WITH_TAX]
注意：此时$作为金额的一部分，不单独标注币种
```

#### 示例3：混淆识别
```
OCR错误：US D
Tokens:  ['US', 'D']
BIO:     [B-CURRENCY, I-CURRENCY]
标注依据：从上下文判断为USD的错误识别
```

### 后处理代码
```python
def normalize_currency(raw_text):
    currency_map = {
        'USD': 'USD', 'US D': 'USD', 'U$D': 'USD',
        'CNY': 'CNY', 'RMB': 'CNY', 
        '美元': 'USD', '人民币': 'CNY',
        '$': 'USD', '￥': 'CNY', 'Y': 'CNY'
    }
    
    cleaned = re.sub(r'[^A-Za-z\$￥€£]', '', raw_text)
    return currency_map.get(cleaned.upper(), 'UNKNOWN')
```

## 4.4 AMOUNT_WITH_TAX / AMOUNT_WITHOUT_TAX / TAX（金额类字段）

### 定义
- AMOUNT_WITH_TAX: 含税总金额
- AMOUNT_WITHOUT_TAX: 不含税金额  
- TAX: 税额

### 常见格式
- 基础：300、300.00、3,000.00
- 带符号：$300、￥300.00
- 全角：３００．００

### 标注原则
- 只标注数字部分和小数点
- 货币符号若贴在数字前，一起标注
- 在表格中根据行标识选择对应金额

### 标注示例

#### 示例1：含税金额
```
OCR原始：Total (incl. tax): 300
Tokens:  ['Total', '(incl.', 'tax):', '300']  
BIO:     [O, O, O, B-AMOUNT_WITH_TAX]
标注依据：根据"Total"和"incl. tax"确定为含税金额
```

#### 示例2：不含税金额
```
OCR原始：Subtotal: 300.00
Tokens:  ['Subtotal:', '300.00']
BIO:     [O, B-AMOUNT_WITHOUT_TAX]  
标注依据：Subtotal通常指不含税金额
```

#### 示例3：多金额干扰
```
发票内容：
Item 1: $100
Item 2: $200  
Subtotal: $300
Tax: $0
Total: $300

只标注汇总行：
'Subtotal:' → O, '$300' → B-AMOUNT_WITHOUT_TAX
'Tax:' → O, '$0' → B-TAX
'Total:' → O, '$300' → B-AMOUNT_WITH_TAX
```

### 后处理代码
```python
def normalize_amount(raw_text):
    # 移除货币符号
    text = re.sub(r'[￥$€£]', '', raw_text)
    # 全角转半角
    text = text.translate(str.maketrans('０１２３４５６７８９．', '0123456789.'))
    # 移除千分位分隔符
    text = text.replace(',', '').replace(' ', '')
    # 转换为浮点数
    try:
        return float(text)
    except ValueError:
        return None  # 需人工审查
```

---

# 5. OCR错误与手写情形处理

## 5.1 标注原则回顾

**关键原则：标注时不纠正OCR输出，但要记录为实体（除非完全不可识别）**

## 5.2 标注流程（步骤化）

### 步骤1：读取OCR输出
- 获取OCR文本和边界框信息
- 如可见，同时查看原始图像确认上下文

### 步骤2：定位候选Token
根据上下文关键词定位：
- 发票号相关：发票号、Invoice No、发票号码、Invoice #
- 日期相关：开票日期、Date、Invoice Date、日期
- 金额相关：合计、Total、Amount、金额、小计、Subtotal
- 币种相关：Currency、币种

### 步骤3：BIO标注
对候选token进行B-I标注，记录不确定情况

### 步骤4：异常标记
完全不可识别的内容标O并加__REVIEW__标记

## 5.3 手写发票处理

### 处理原则
- 优先基于OCR输出进行标注
- OCR完全错误时标注为O，在注释中记录人工识别值
- 将手写样本归入"人工校验集"供后续处理

### 示例：手写发票处理
```
原始图像：手写发票号 Y309824263008
OCR输出：Y30982426JOO8（J、O为手写识别错误）

标注：
Tokens: ['Y30982426JOO8']
BIO:    [B-INVOICE_NO]
注释：   __MANUAL_CHECK__ 手写识别，实际值可能为Y309824263008
```

## 5.4 常见OCR错误示例

### 错误类型1：字符混淆
```
正确: Y 309824263008
OCR:  Y 3O9824263OO8  (0→O混淆)
标注: B-INVOICE_NO I-INVOICE_NO
处理: 标注为实体，后处理纠正
```

### 错误类型2：空格异常
```
正确: 2025年6月30日
OCR:  2025年 6 月30日  (空格错插)
标注: B-DATE I-DATE I-DATE
处理: 保持实体连续性
```

### 错误类型3：跨行分割
```
正确: Total Amount: 300.00
OCR:  Total Amount: 300.
      00
标注: O O B-AMOUNT_WITH_TAX I-AMOUNT_WITH_TAX
处理: 跨行仍为同一实体
```

### 错误类型4：符号混淆
```
正确: $300
OCR:  S300  ($→S)
标注: B-AMOUNT_WITH_TAX
处理: 根据上下文推断为金额
```

### 错误类型5：数字粘连
```
正确: Invoice No: Y309824263008
OCR:  InvoiceNo:Y309824263008  (空格丢失)
标注: O B-INVOICE_NO
处理: 基于OCR输出的token划分
```

### 错误类型6：中英混合错误
```
正确: 金额：￥300.00
OCR:  金額:Y300.OO  (额→額, ￥→Y, 0→O)
标注: O O B-AMOUNT_WITH_TAX
处理: 标注实体，注释原文语言
```

### 错误类型7：完全乱码
```
OCR:  ###$@%!
标注: O O O
注释: __REVIEW__ 完全不可识别，需查看原图
```

### 错误类型8：部分可识别
```
OCR:  Y3O98###63OO8  (中间乱码)
标注: B-INVOICE_NO I-INVOICE_NO I-INVOICE_NO
注释: __PARTIAL_OCR__ 部分字符不清，但可识别为发票号
```

---

# 6. 表格与行项目处理

## 6.1 标注范围

**只标注头部/汇总信息，不标注商品明细行**

标注内容：
- ✅ 发票号码、开票日期、币种
- ✅ 合计金额、不含税金额、税额
- ❌ 商品名称、单价、数量等明细

## 6.2 多金额选择优先级

当表格中出现多个金额时，按以下优先级选择：

### 优先级1：明确文本标识
- "合计"、"Total"、"Invoice Total"
- "Amount Due"、"应付金额"
- "Grand Total"、"总计"

### 优先级2：位置特征
- 表格底部位置
- 汇总行位置
- 加粗或突出显示

### 优先级3：数值特征
- 数值最大且合理
- 带货币符号
- 格式完整（含小数点）

## 6.3 表格处理示例

### 示例：复杂表格
```
发票内容：
+------------------------+--------+
| Item Description       | Amount |  
+------------------------+--------+
| Product A             | 100.00 |
| Product B             | 200.00 |  
+------------------------+--------+
| Subtotal              | 300.00 |
| Tax (0%)              |   0.00 |
| Invoice Total         | 300.00 |
+------------------------+--------+

标注策略：
- 忽略 Product A: 100.00, Product B: 200.00
- 标注 Subtotal: 300.00 → B-AMOUNT_WITHOUT_TAX  
- 标注 Tax (0%): 0.00 → B-TAX
- 标注 Invoice Total: 300.00 → B-AMOUNT_WITH_TAX
```

### 示例：多币种表格
```
发票内容：
Service Fee: $50
Tax: $10
Total (USD): $60
Total (CNY): ¥420

标注策略：
- 根据主要币种选择USD金额
- 标注Total (USD): $60 → B-AMOUNT_WITH_TAX
- 标注USD → B-CURRENCY
- 忽略CNY换算金额
```

## 6.4 歧义处理

当无法确定正确金额时：
1. 标注所有可能的候选金额
2. 添加__AMBIGUOUS__标记
3. 在注释中说明歧义原因
4. 提交给标注主管仲裁

**示例：**
```
发票中同时出现：
Net Amount: 300
Total Amount: 300  
Final Total: 300

处理：都标注为相应实体类型，添加注释
```

---

# 7. 标注工具配置

## 7.1 Doccano配置

### 项目设置
1. 创建新项目 → 选择"Sequence Labeling"
2. 设置标签：

```json
[
  {"text": "INVOICE_NO", "prefix_key": "1", "background_color": "#FF6B6B"},
  {"text": "DATE", "prefix_key": "2", "background_color": "#4ECDC4"}, 
  {"text": "CURRENCY", "prefix_key": "3", "background_color": "#45B7D1"},
  {"text": "AMOUNT_WITH_TAX", "prefix_key": "4", "background_color": "#96CEB4"},
  {"text": "AMOUNT_WITHOUT_TAX", "prefix_key": "5", "background_color": "#FFEAA7"},
  {"text": "TAX", "prefix_key": "6", "background_color": "#DDA0DD"}
]
```

### UI操作步骤
1. 导入OCR文本文件
2. 选择"字符级选择"模式（推荐）
3. 按住鼠标拖选token范围
4. 按快捷键（1-6）快速标注
5. 在Comments字段添加__REVIEW__等标记

### 优缺点
- ✅ 支持字符级精确选择
- ✅ 快捷键操作效率高  
- ❌ 不支持边界框可视化

## 7.2 Label Studio配置

### JSON配置示例
```xml
<View>
  <Text name="text" value="$ocr_text"/>
  <Labels name="label" toName="text">
    <Label value="INVOICE_NO" background="red"/>
    <Label value="DATE" background="blue"/>
    <Label value="CURRENCY" background="green"/>
    <Label value="AMOUNT_WITH_TAX" background="orange"/> 
    <Label value="AMOUNT_WITHOUT_TAX" background="purple"/>
    <Label value="TAX" background="brown"/>
  </Labels>
  <TextArea name="notes" toName="text" 
           placeholder="标注备注，如__REVIEW__"/>
</View>
```

### 设置步骤
1. 创建项目 → 选择"Named Entity Recognition"
2. 导入配置JSON
3. 导入数据（支持与图像关联）
4. 开始标注

### 优缺点
- ✅ 支持图像+文本联合显示
- ✅ 支持边界框可视化
- ❌ 字符级选择相对复杂

## 7.3 工具选择建议

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| 纯文本标注 | Doccano | 字符级选择便捷 |
| 多模态标注 | Label Studio | 支持图像+文本 |
| 大规模标注 | Doccano | 批量操作效率高 |
| 质量审查 | Label Studio | 可视化便于审查 |

---

# 8. 输出格式与后处理

## 8.1 训练所需输出格式

### 标准JSON格式
```json
{
  "text": "发票号码: Y 309824263008 开票日期: 2025年6月30日 合计: $300",
  "tokens": ["发票号码", ":", "Y", "309824263008", "开票日期", ":", "2025年6月30日", "合计", ":", "$", "300"],
  "labels": ["O", "O", "B-INVOICE_NO", "I-INVOICE_NO", "O", "O", "B-DATE", "O", "O", "B-CURRENCY", "B-AMOUNT_WITH_TAX"],
  "entities": [
    {
      "start": 2, "end": 4, "label": "INVOICE_NO", 
      "text": "Y 309824263008", "confidence": 0.9
    },
    {
      "start": 6, "end": 7, "label": "DATE",
      "text": "2025年6月30日", "confidence": 0.8
    }
  ],
  "bboxes": [
    [100, 50, 200, 70], [201, 50, 210, 70], 
    [220, 50, 240, 70], [241, 50, 350, 70]
  ],
  "image_path": "data/images/invoice_001.jpg",
  "normalized": {
    "INVOICE_NO": "Y309824263008",
    "DATE": "2025-06-30", 
    "CURRENCY": "USD",
    "AMOUNT_WITH_TAX": 300.0
  }
}
```

### CoNLL格式
```
发	O
票	O  
号	O
码	O
:	O
Y	B-INVOICE_NO
309824263008	I-INVOICE_NO

开	O
票	O
日	O
期	O
:	O
2025年6月30日	B-DATE
```

## 8.2 字段归一化流程

### 发票号归一化
```python
def normalize_invoice_no(tokens, labels):
    # 提取实体tokens
    entity_tokens = extract_entity_tokens(tokens, labels, 'INVOICE_NO')
    raw_text = ''.join(entity_tokens)
    
    # 清理和标准化
    normalized = raw_text.upper()
    normalized = re.sub(r'[^A-Z0-9\-]', '', normalized) 
    normalized = normalized.replace('O', '0').replace('I', '1')
    
    return normalized
```

### 日期归一化
```python
def normalize_date(tokens, labels):
    entity_tokens = extract_entity_tokens(tokens, labels, 'DATE')
    raw_text = ''.join(entity_tokens)
    
    # 中文日期处理
    match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', raw_text)
    if match:
        y, m, d = match.groups()
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    
    # 其他格式尝试解析
    try:
        from dateutil import parser
        parsed = parser.parse(raw_text, fuzzy=True)
        return parsed.strftime('%Y-%m-%d')
    except:
        return None
```

### 金额归一化
```python
def normalize_amount(tokens, labels, amount_type):
    entity_tokens = extract_entity_tokens(tokens, labels, amount_type)
    raw_text = ''.join(entity_tokens)
    
    # 移除货币符号和分隔符
    cleaned = re.sub(r'[$￥€£,\s]', '', raw_text)
    cleaned = cleaned.replace('，', '')  # 中文逗号
    
    try:
        return float(cleaned)
    except ValueError:
        return None
```

## 8.3 校验规则

### 自动校验脚本
```python
def validate_extraction(result):
    errors = []
    
    # 发票号长度检查
    if 'INVOICE_NO' in result:
        if len(result['INVOICE_NO']) < 6 or len(result['INVOICE_NO']) > 30:
            errors.append("发票号长度异常")
    
    # 日期有效性检查 
    if 'DATE' in result:
        try:
            datetime.strptime(result['DATE'], '%Y-%m-%d')
        except:
            errors.append("日期格式无效")
    
    # 金额逻辑检查
    if all(k in result for k in ['AMOUNT_WITH_TAX', 'AMOUNT_WITHOUT_TAX', 'TAX']):
        expected_total = result['AMOUNT_WITHOUT_TAX'] + result['TAX']
        if abs(expected_total - result['AMOUNT_WITH_TAX']) > 0.01:
            errors.append("金额关系不符：含税金额 ≠ 不含税金额 + 税额")
    
    # 币种检查
    valid_currencies = ['USD', 'CNY', 'EUR', 'JPY', 'GBP']
    if 'CURRENCY' in result and result['CURRENCY'] not in valid_currencies:
        errors.append(f"不支持的币种: {result['CURRENCY']}")
    
    return errors
```

---

# 9. 质量保证与标注一致性检查

## 9.1 双标策略

### 实施方案
- **比例**: 对10%的样本进行双标（两名标注员独立标注）
- **选择**: 优先选择复杂样本（手写、多语言、OCR错误严重）
- **时间**: 项目初期20%双标，中期10%，后期5%

### 一致性计算
```python
def calculate_entity_f1(annotations1, annotations2):
    # 实体级别F1计算
    entities1 = extract_entities(annotations1)
    entities2 = extract_entities(annotations2) 
    
    tp = len(entities1 & entities2)  # 完全匹配
    fp = len(entities1 - entities2)  
    fn = len(entities2 - entities1)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0  
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_cohen_kappa(annotations1, annotations2):
    # 计算Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score
    labels1 = [ann['labels'] for ann in annotations1]
    labels2 = [ann['labels'] for ann in annotations2]
    return cohen_kappa_score(labels1, labels2)
```

### 质量阈值
| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| 实体F1 | >0.9 | 0.8-0.9 | <0.8 |
| Cohen's Kappa | >0.8 | 0.6-0.8 | <0.6 |
| Token准确率 | >0.95 | 0.9-0.95 | <0.9 |

## 9.2 常见冲突场景与仲裁规则

### 冲突场景1：边界争议
```
文本: "Invoice Total: $300.00"
标注员A: ['Invoice', 'Total:', '$300.00'] → [O, O, B-AMOUNT_WITH_TAX]
标注员B: ['Invoice', 'Total:', '$', '300.00'] → [O, O, B-CURRENCY, B-AMOUNT_WITH_TAX]

仲裁规则: 优先整体标注，$300.00作为完整金额实体
```

### 冲突场景2：实体类型争议
```
文本: "Net Amount: 300"  
标注员A: B-AMOUNT_WITHOUT_TAX
标注员B: B-AMOUNT_WITH_TAX

仲裁规则: 查看上下文，"Net"通常指不含税，选择A
```

### 冲突场景3：OCR错误处理争议
```
OCR: "Y 3O982426JOO8"
标注员A: B-INVOICE_NO I-INVOICE_NO (标注为实体)
标注员B: O O (标注为非实体，认为错误太多)

仲裁规则: 优先标注为实体，后处理阶段纠正
```

## 9.3 标注审查检查清单

### 每张发票必检项目
- [ ] 发票号是否存在且唯一？
- [ ] 日期格式是否识别正确？
- [ ] 金额标注是否在合理范围？
- [ ] 币种是否在支持列表中？
- [ ] 含税/不含税/税额关系是否合理？
- [ ] 跨行实体是否保持连续？
- [ ] __REVIEW__标记是否恰当？

### 批量检查脚本
```python
def batch_quality_check(annotations):
    issues = []
    
    for i, ann in enumerate(annotations):
        # 检查必需字段
        required_entities = ['INVOICE_NO', 'DATE'] 
        missing = [e for e in required_entities if not has_entity(ann, e)]
        if missing:
            issues.append(f"样本{i}: 缺少必需字段 {missing}")
        
        # 检查标注一致性
        if has_inconsistent_bio(ann):
            issues.append(f"样本{i}: BIO序列不一致")
            
        # 检查异常长度
        for entity in get_entities(ann):
            if entity['type'] == 'INVOICE_NO' and len(entity['text']) > 50:
                issues.append(f"样本{i}: 发票号过长")
    
    return issues
```

---

# 10. 标注示例集合

基于提供的示例数据，生成多种标注场景的完整示例：

```json
{
  "image": "data/converted_images/invoice_page_1.jpg",
  "ground_truth": {
    "InvoiceNo": "Y 309824263008",
    "InvoiceDate": "2025年6月30日", 
    "Currency": "USD",
    "Amount with Tax": "300",
    "Amount without Tax": "300",
    "Tax": "0"
  }
}
```

## 10.1 理想正确OCR情况

### OCR输出文本
```
发票号码: Y 309824263008
开票日期: 2025年6月30日
币种: USD
不含税金额: 300
税额: 0  
含税金额: 300
```

### Tokenization与标注
| Token | Label |
|--------|-------|
| 发票号码 | O |
| : | O |  
| Y | B-INVOICE_NO |
| 309824263008 | I-INVOICE_NO |
| 开票日期 | O |
| : | O |
| 2025年6月30日 | B-DATE |
| 币种 | O |
| : | O |
| USD | B-CURRENCY |
| 不含税金额 | O |
| : | O |
| 300 | B-AMOUNT_WITHOUT_TAX |
| 税额 | O |
| : | O |
| 0 | B-TAX |
| 含税金额 | O |
| : | O |
| 300 | B-AMOUNT_WITH_TAX |

### 标注依据
- 基于关键词"发票号码"识别Y 309824263008为发票号
- "开票日期"后的完整日期标注为DATE实体
- "币种"明确标识USD为货币类型
- 根据"不含税金额"、"税额"、"含税金额"分别标注对应数值

### 最终JSON输出
```json
{
  "tokens": ["发票号码", ":", "Y", "309824263008", "开票日期", ":", "2025年6月30日", "币种", ":", "USD", "不含税金额", ":", "300", "税额", ":", "0", "含税金额", ":", "300"],
  "labels": ["O", "O", "B-INVOICE_NO", "I-INVOICE_NO", "O", "O", "B-DATE", "O", "O", "B-CURRENCY", "O", "O", "B-AMOUNT_WITHOUT_TAX", "O", "O", "B-TAX", "O", "O", "B-AMOUNT_WITH_TAX"],
  "normalized": {
    "INVOICE_NO": "Y309824263008",
    "DATE": "2025-06-30",
    "CURRENCY": "USD", 
    "AMOUNT_WITHOUT_TAX": 300.0,
    "TAX": 0.0,
    "AMOUNT_WITH_TAX": 300.0
  }
}
```

## 10.2 常见OCR错误情况

### OCR错误输出
```
发票号码: Y 3O98242 63OO8
开票曰期: 2O25年6月JO曰
币种: US D
不含税金額: JOO
税額: O
含税金額: JOO
```

### 错误分析
- 数字0→字母O：3O98242, 63OO8, 2O25, JO, JOO  
- 字母混淆：曰→日, US D→USD, 額→额
- 空格错插：Y 3O98242 63OO8中多余空格

### Tokenization与标注
| Token | Label | 备注 |
|--------|-------|------|
| 发票号码 | O | |
| : | O | |
| Y | B-INVOICE_NO | |
| 3O98242 | I-INVOICE_NO | 含OCR错误但仍标注 |
| 63OO8 | I-INVOICE_NO | 跨空格连续实体 |
| 开票曰期 | O | 曰→日OCR错误 |
| : | O | |
| 2O25年6月JO曰 | B-DATE | 含多个OCR错误 |
| 币种 | O | |
| : | O | |
| US | B-CURRENCY | USD被分割 |
| D | I-CURRENCY | |
| 不含税金額 | O | 額→额错误 |
| : | O | |
| JOO | B-AMOUNT_WITHOUT_TAX | J→3, O→0错误 |
| 税額 | O | |
| : | O | |
| O | B-TAX | O→0错误 |
| 含税金額 | O | |
| : | O | |
| JOO | B-AMOUNT_WITH_TAX | |

### 标注依据
- 虽有大量OCR错误，仍基于上下文关键词判断实体类型
- 保持实体连续性，即使被错误分割
- 在后处理阶段纠正字符错误，标注阶段忠于OCR输出

### 最终JSON输出
```json
{
  "tokens": ["发票号码", ":", "Y", "3O98242", "63OO8", "开票曰期", ":", "2O25年6月JO曰", "币种", ":", "US", "D", "不含税金額", ":", "JOO", "税額", ":", "O", "含税金額", ":", "JOO"],
  "labels": ["O", "O", "B-INVOICE_NO", "I-INVOICE_NO", "I-INVOICE_NO", "O", "O", "B-DATE", "O", "O", "B-CURRENCY", "I-CURRENCY", "O", "O", "B-AMOUNT_WITHOUT_TAX", "O", "O", "B-TAX", "O", "O", "B-AMOUNT_WITH_TAX"],
  "normalized": {
    "INVOICE_NO": "Y309824263008",  
    "DATE": "2025-06-30",
    "CURRENCY": "USD",
    "AMOUNT_WITHOUT_TAX": 300.0,
    "TAX": 0.0, 
    "AMOUNT_WITH_TAX": 300.0
  },
  "ocr_confidence": "low",
  "review_notes": "大量OCR错误，需人工验证"
}
```

## 10.3 跨行切割情况

### OCR跨行输出
```
发票号码: Y 309824
263008
开票日期: 2025年6月
30日
合计金额: 
$300
```

### Tokenization与标注
| Token | Label | 备注 |
|--------|-------|------|  
| 发票号码 | O | |
| : | O | |
| Y | B-INVOICE_NO | |
| 309824 | I-INVOICE_NO | 第一行 |
| 263008 | I-INVOICE_NO | 第二行，保持连续 |
| 开票日期 | O | |
| : | O | |
| 2025年6月 | B-DATE | 跨行日期开始 |
| 30日 | I-DATE | 跨行日期结束 |
| 合计金额 | O | |
| : | O | |
| $ | B-AMOUNT_WITH_TAX | 货币符号 |
| 300 | I-AMOUNT_WITH_TAX | 金额数字 |

### 标注依据
- 跨行但语义连续的实体保持B-I标注序列
- 发票号虽分两行显示，仍为同一实体
- 日期跨行时保持连续标注
- 货币符号与数字虽分行但属于同一金额实体

## 10.4 手写OCR误识别

### 手写发票OCR输出
```  
发票号碼: Y JO98Z426JOO8  (手写导致多处错误)
开票曰期: 20Z5年6月JO曰
今额: 手写JOO美元
```

### Tokenization与标注
| Token | Label | 备注 |
|--------|-------|------|
| 发票号碼 | O | 碼→码繁简混用 |
| : | O | |
| Y | B-INVOICE_NO | |
| JO98Z426JOO8 | I-INVOICE_NO | 手写严重错误 |
| 开票曰期 | O | |
| : | O | |
| 20Z5年6月JO曰 | B-DATE | 手写Z、J、曰错误 |
| 今额 | O | 金→今手写错误 |
| : | O | |
| 手写JOO美元 | B-AMOUNT_WITH_TAX | OCR识别出"手写"字样 |

### 特殊处理
```json
{
  "review_flags": ["__HANDWRITTEN__", "__MANUAL_CHECK__"],
  "manual_interpretation": {
    "INVOICE_NO": "Y309824263008",  
    "DATE": "2025年6月30日",
    "AMOUNT_WITH_TAX": "300美元"
  },
  "confidence": "very_low",
  "action_required": "人工校验集"
}
```

## 10.5 多语言变体

### 中文发票示例
```
发票号: Y 309824263008  
开票日期: 二〇二五年六月三十日
币种: 美元
合计: 三百元整
```

### 英文发票示例  
```
Invoice No: Y 309824263008
Invoice Date: June 30, 2025
Currency: USD  
Total Amount: $300.00
```

### 混合中英发票示例
```
Invoice发票号: Y 309824263008
Date开票日期: 2025年June月30日  
Currency币种: USD美元
Total合计: $300.00元
```

## 10.6 表格多金额干扰

### 复杂表格OCR输出
```
产品A        数量2    单价150    小计300
产品B        数量0    单价200    小计0
                    小计Subtotal: 300
                    税率Tax Rate: 0%  
                    税额Tax: 0
                    合计Total: 300
发票号Invoice No: Y 309824263008
```

### 标注策略
只标注汇总行，忽略明细：

| Token | Label | 备注 |
|--------|-------|------|
| 产品A | O | 忽略明细 |
| ... | O | 忽略中间明细行 |
| 小计Subtotal | O | |
| : | O | |
| 300 | B-AMOUNT_WITHOUT_TAX | 不含税小计 |
| 税额Tax | O | |
| : | O | |
| 0 | B-TAX | |
| 合计Total | O | |
| : | O | |
| 300 | B-AMOUNT_WITH_TAX | 含税合计 |
| 发票号Invoice | O | |
| No | O | |
| : | O | |
| Y | B-INVOICE_NO | |
| 309824263008 | I-INVOICE_NO | |

---

# 11. 快速查阅页 (Cheat Sheet)

## 11.1 实体类型速查

| 标签 | 中文含义 | 关键词提示 | 示例值 |
|------|----------|-----------|--------|
| INVOICE_NO | 发票号码 | 发票号、Invoice No | Y309824263008 |
| DATE | 开票日期 | 开票日期、Invoice Date | 2025-06-30 |
| CURRENCY | 币种 | 币种、Currency | USD |
| AMOUNT_WITH_TAX | 含税金额 | 合计、Total | 300.00 |
| AMOUNT_WITHOUT_TAX | 不含税金额 | 小计、Subtotal | 300.00 |
| TAX | 税额 | 税额、Tax | 0.00 |

## 11.2 常见OCR混淆对照表

| 正确 | 常见错误 | 正确 | 常见错误 |
|------|----------|------|----------|
| 0 | O, o, Q | 1 | I, l, 丨 |
| 2 | Z, z | 5 | S, s |
| 6 | G, b | 8 | B, 3 |
| $ | S, 5 | ￥ | Y, y |
| 日 | 曰, 目 | 额 | 額, 頟 |

## 11.3 快速决策流程

```
遇到token时：
1. 查看前后是否有关键词？ → 是：标注对应实体
2. 是否在表格汇总区域？ → 是：重点关注  
3. 是否为数字+货币符号？ → 是：可能为金额
4. 是否为日期格式？ → 是：标注DATE
5. 完全不可识别？ → 标注O + __REVIEW__
6. 部分可识别？ → 标注实体 + 注释
```

## 11.4 常用正则表达式

```python
# 发票号
r'[A-Za-z0-9\-\s]{6,30}'

# 日期(中文)  
r'\d{4}年\d{1,2}月\d{1,2}日'

# 日期(数字)
r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'

# 金额
r'\d+\.?\d*'

# 币种代码
r'[A-Z]{3}'
```

## 11.5 遇到问题怎么办

| 情况 | 处理方式 | 标记 |
|------|---------|------|
| 完全无法识别 | 标注O | __REVIEW__ |
| OCR错误明显 | 标注实体 | __OCR_ERROR__ |
| 边界不确定 | 标注较大范围 | __BOUNDARY__ |
| 多个候选值 | 都标注 | __AMBIGUOUS__ |
| 手写难识别 | 标注OCR输出 | __HANDWRITTEN__ |
| 跨页面实体 | 分别标注 | __CROSS_PAGE__ |

## 11.6 质量检查口诀

**每张发票六必查：**
1. 号码有没有？（发票号）
2. 日期对不对？（开票日期）  
3. 币种在不在？（货币类型）
4. 金额算不算？（数学关系）
5. 跨行连不连？（实体连续）
6. 标记全不全？（异常标记）

---

# 12. 附录：常用正则与代码片段

## 12.1 字段验证正则

### 发票号验证
```python
import re

def validate_invoice_no(text):
    # 基本格式检查：6-30位字母数字组合
    if not re.match(r'^[A-Za-z0-9\-]{6,30}$', text):
        return False, "格式不符：需要6-30位字母数字"
    
    # 过度简单检查
    if re.match(r'^[0-9]+$', text) and len(text) < 8:
        return False, "疑似纯数字过短"
        
    return True, "验证通过"

# 示例使用
valid, msg = validate_invoice_no("Y309824263008")
print(f"Y309824263008: {msg}")  # 验证通过
```

### 日期解析与验证
```python
import re
from datetime import datetime
from dateutil import parser

def parse_date(text):
    """多格式日期解析"""
    # 中文格式：2025年6月30日
    chinese_pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})日'
    match = re.search(chinese_pattern, text)
    if match:
        year, month, day = match.groups()
        try:
            date_obj = datetime(int(year), int(month), int(day))
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            return None
    
    # 标准格式尝试
    try:
        parsed = parser.parse(text, fuzzy=True)
        return parsed.strftime('%Y-%m-%d')
    except:
        return None

# 示例
print(parse_date("2025年6月30日"))     # 2025-06-30
print(parse_date("June 30, 2025"))    # 2025-06-30  
print(parse_date("30/06/2025"))       # 2025-06-30
```

### 币种标准化
```python
def normalize_currency(text):
    """币种标准化映射"""
    currency_mapping = {
        # 代码形式
        'USD': 'USD', 'US D': 'USD', 'U$D': 'USD',
        'CNY': 'CNY', 'RMB': 'CNY', 'CN Y': 'CNY',
        'EUR': 'EUR', 'EU R': 'EUR',
        'JPY': 'JPY', 'JP Y': 'JPY',
        'GBP': 'GBP', 'GB P': 'GBP',
        
        # 符号形式
        '$': 'USD', 'S': 'USD',  # $ OCR错误为S
        '￥': 'CNY', 'Y': 'CNY', '¥': 'CNY',
        '€': 'EUR', 'E': 'EUR',
        '£': 'GBP', 'L': 'GBP',
        
        # 中文形式
        '美元': 'USD', '美金': 'USD',
        '人民币': 'CNY', '元': 'CNY',
        '欧元': 'EUR',
        '日元': 'JPY', '日币': 'JPY',
        '英镑': 'GBP'
    }
    
    cleaned = text.strip().upper()
    return currency_mapping.get(cleaned, 'UNKNOWN')

# 示例
print(normalize_currency("US D"))  # USD
print(normalize_currency("美元"))   # USD
print(normalize_currency("Y"))     # CNY
```

### 金额清理与转换
```python
def clean_amount(text):
    """金额文本清理和数值转换"""
    # 移除货币符号
    cleaned = re.sub(r'[￥$€£]', '', text)
    
    # 全角转半角
    full_to_half = str.maketrans(
        '０１２３４５６７８９．，',
        '0123456789.,')
    cleaned = cleaned.translate(full_to_half)
    
    # 移除千位分隔符和空格
    cleaned = cleaned.replace(',', '').replace(' ', '')
    
    # 处理中文数字（简单版本）
    chinese_digits = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                     '六': '6', '七': '7', '八': '8', '九': '9', '零': '0',
                     '十': '10', '百': '100', '千': '1000', '万': '10000'}
    
    for cn, num in chinese_digits.items():
        if cn in cleaned:
            # 简单替换，实际使用需要更复杂的中文数字解析
            cleaned = cleaned.replace(cn, num)
    
    # 转换为浮点数
    try:
        return float(cleaned)
    except ValueError:
        return None

# 示例
print(clean_amount("$3,000.50"))   # 3000.5
print(clean_amount("￥１２３４"))    # 1234.0
print(clean_amount("三百"))         # 300.0 (简化处理)
```

## 12.2 BIO序列处理工具

### 实体提取函数
```python
def extract_entities_from_bio(tokens, labels):
    """从BIO序列中提取实体"""
    entities = []
    current_entity = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # 保存前一个实体
            if current_entity:
                entities.append(current_entity)
            
            # 开始新实体
            entity_type = label[2:]  # 去掉 'B-' 前缀
            current_entity = {
                'type': entity_type,
                'tokens': [token],
                'start': i,
                'end': i + 1
            }
            
        elif label.startswith('I-') and current_entity:
            # 继续当前实体
            entity_type = label[2:]
            if entity_type == current_entity['type']:
                current_entity['tokens'].append(token)
                current_entity['end'] = i + 1
            else:
                # 类型不匹配，结束当前实体
                entities.append(current_entity)
                current_entity = None
                
        else:
            # O标签或序列错误，结束当前实体
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # 处理最后一个实体
    if current_entity:
        entities.append(current_entity)
    
    # 合并token文本
    for entity in entities:
        entity['text'] = ''.join(entity['tokens'])
    
    return entities

# 示例使用
tokens = ['发票号码', ':', 'Y', '309824263008', '金额', ':', '300']
labels = ['O', 'O', 'B-INVOICE_NO', 'I-INVOICE_NO', 'O', 'O', 'B-AMOUNT_WITH_TAX']

entities = extract_entities_from_bio(tokens, labels)
for entity in entities:
    print(f"{entity['type']}: {entity['text']} (位置: {entity['start']}-{entity['end']})")
```

### BIO序列验证
```python
def validate_bio_sequence(labels):
    """验证BIO序列的正确性"""
    errors = []
    
    for i, label in enumerate(labels):
        if label.startswith('I-'):
            # I标签前面应该有对应的B或I标签
            if i == 0:
                errors.append(f"位置{i}: I标签不能出现在开头")
            else:
                prev_label = labels[i-1]
                current_type = label[2:]
                
                if prev_label == 'O':
                    errors.append(f"位置{i}: I-{current_type}前面不能是O")
                elif prev_label.startswith('B-') or prev_label.startswith('I-'):
                    prev_type = prev_label[2:]
                    if prev_type != current_type:
                        errors.append(f"位置{i}: I-{current_type}与前面的{prev_label}类型不匹配")
    
    return errors

# 示例验证
labels1 = ['O', 'B-INVOICE_NO', 'I-INVOICE_NO', 'O']  # 正确
labels2 = ['I-INVOICE_NO', 'O', 'B-DATE']            # 错误：开头就是I
labels3 = ['B-INVOICE_NO', 'I-DATE', 'O']            # 错误：类型不匹配

print("序列1错误:", validate_bio_sequence(labels1))  # []
print("序列2错误:", validate_bio_sequence(labels2))  # [错误信息]
print("序列3错误:", validate_bio_sequence(labels3))  # [错误信息]
```

## 12.3 批量处理脚本模板

### 标注结果后处理管道
```python
class InvoiceAnnotationProcessor:
    def __init__(self):
        self.currency_normalizer = normalize_currency
        self.amount_cleaner = clean_amount
        self.date_parser = parse_date
        
    def process_single_annotation(self, annotation):
        """处理单个标注结果"""
        tokens = annotation['tokens']
        labels = annotation['labels']
        
        # 提取实体
        entities = extract_entities_from_bio(tokens, labels)
        
        # 归一化各字段
        normalized = {}
        for entity in entities:
            entity_type = entity['type']
            raw_text = entity['text']
            
            if entity_type == 'INVOICE_NO':
                normalized[entity_type] = self.normalize_invoice_no(raw_text)
            elif entity_type == 'DATE':
                normalized[entity_type] = self.date_parser(raw_text)
            elif entity_type == 'CURRENCY':
                normalized[entity_type] = self.currency_normalizer(raw_text)
            elif entity_type.startswith('AMOUNT') or entity_type == 'TAX':
                normalized[entity_type] = self.amount_cleaner(raw_text)
        
        # 验证逻辑关系
        validation_errors = self.validate_amounts(normalized)
        
        return {
            'original': annotation,
            'entities': entities,
            'normalized': normalized,
            'validation_errors': validation_errors
        }
    
    def normalize_invoice_no(self, text):
        # 实现发票号归一化逻辑
        return normalize_invoice_no(text)
    
    def validate_amounts(self, normalized):
        """验证金额逻辑关系"""
        errors = []
        
        required_keys = ['AMOUNT_WITH_TAX', 'AMOUNT_WITHOUT_TAX', 'TAX']
        if all(key in normalized for key in required_keys):
            with_tax = normalized['AMOUNT_WITH_TAX']
            without_tax = normalized['AMOUNT_WITHOUT_TAX']
            tax = normalized['TAX']
            
            if abs((without_tax + tax) - with_tax) > 0.01:
                errors.append(f"金额关系错误: {without_tax} + {tax} ≠ {with_tax}")
        
        return errors

# 使用示例
processor = InvoiceAnnotationProcessor()

sample_annotation = {
    'tokens': ['发票号', ':', 'Y', '309824263008', '合计', ':', '300'],
    'labels': ['O', 'O', 'B-INVOICE_NO', 'I-INVOICE_NO', 'O', 'O', 'B-AMOUNT_WITH_TAX']
}

result = processor.process_single_annotation(sample_annotation)
print("处理结果:", result['normalized'])
```

---

# 13. 标注员检查清单与质检计划

## 13.1 标注员每日检查清单

### A. 标注前准备
- [ ] 确认OCR文本与图像对应
- [ ] 检查tokenization是否合理  
- [ ] 核实标注工具配置正确

### B. 标注中检查
- [ ] 每个实体都有明确的B开头标记
- [ ] I标记与前面的B标记类型一致
- [ ] 跨行实体保持连续标注
- [ ] 完全不可识别内容标为O并加备注
- [ ] OCR错误内容仍标注为实体（不纠正）

### C. 标注后验证
- [ ] 必需字段：发票号、日期是否标注
- [ ] 金额字段：至少标注一个金额类型
- [ ] BIO序列：没有孤立的I标记
- [ ] 特殊标记：__REVIEW__等使用恰当
- [ ] 边界检查：实体边界选择合理

## 13.2 质检采样计划

### 阶段性采样比例
| 项目阶段 | 双标比例 | 抽检比例 | 重点检查内容 |
|----------|----------|----------|-------------|
| 启动期(前100份) | 20% | 50% | 理解规范，建立基准 |
| 稳定期(100-1000份) | 10% | 20% | 一致性监控 |
| 成熟期(1000份+) | 5% | 10% | 质量维持 |

### 重点抽检样本
- **复杂样本**：手写、多语言、严重OCR错误
- **边界案例**：金额为0、币种缺失、日期异常  
- **新标注员**：前50份100%检查
- **争议样本**：双标不一致的情况

### Kappa计算建议
```python
def calculate_annotation_quality(annotations_batch):
    """计算标注质量指标"""
    from sklearn.metrics import cohen_kappa_score, classification_report
    
    # 假设有两名标注员的结果
    annotator1_labels = []
    annotator2_labels = []
    
    for sample in annotations_batch:
        annotator1_labels.extend(sample['labels_1'])
        annotator2_labels.extend(sample['labels_2'])
    
    # 计算Kappa值  
    kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
    
    # 计算详细报告
    report = classification_report(annotator1_labels, annotator2_labels, 
                                 target_names=['O', 'B-INVOICE_NO', 'I-INVOICE_NO', 
                                             'B-DATE', 'I-DATE', 'B-CURRENCY', 
                                             'B-AMOUNT_WITH_TAX', 'B-AMOUNT_WITHOUT_TAX', 'B-TAX'])
    
    return {
        'kappa': kappa,
        'quality_level': 'excellent' if kappa > 0.8 else 'good' if kappa > 0.6 else 'needs_improvement',
        'detailed_report': report
    }
```

### 质量改进行动
| Kappa范围 | 质量等级 | 行动建议 |
|-----------|----------|---------|
| > 0.8 | 优秀 | 保持当前标准 |
| 0.6-0.8 | 良好 | 重点培训难点 |
| 0.4-0.6 | 一般 | 增加双标比例，规范培训 |
| < 0.4 | 较差 | 暂停标注，全员重训 |

## 13.3 最终交付检查单

### 数据集完整性检查
- [ ] 样本总数符合预期
- [ ] 所有必需字段都有标注覆盖
- [ ] 文件格式符合训练要求
- [ ] 归一化结果格式统一

### 质量指标达标检查  
- [ ] 双标样本Kappa > 0.7
- [ ] 实体级F1 > 0.85
- [ ] 必需字段召回率 > 0.9
- [ ] 异常标记使用恰当

### 技术规范检查
- [ ] BIO序列格式正确
- [ ] JSON输出格式符合规范
- [ ] 字符编码统一(UTF-8)
- [ ] 归一化代码可执行