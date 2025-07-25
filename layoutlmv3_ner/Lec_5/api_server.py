from flask import Flask, request, jsonify
from src.inference_pipeline import InvoiceInferencePipeline

app = Flask(__name__)
pipeline = InvoiceInferencePipeline("./models/final_invoice_layoutlmv3", "./data")

@app.route('/extract_invoice', methods=['POST'])
def extract_invoice():
    """发票信息提取API"""
    try:
        file = request.files['invoice']
        if file:
            # 保存临时文件
            temp_path = f"./temp/{file.filename}"
            file.save(temp_path)
            
            # 处理发票
            result = pipeline.process_invoice(temp_path)
            
            # 清理临时文件
            os.remove(temp_path)
            
            return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)