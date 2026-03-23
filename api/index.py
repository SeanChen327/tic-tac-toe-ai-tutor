from http.server import BaseHTTPRequestHandler
import json
import os
import requests
from google import genai

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html_path = os.path.join(os.path.dirname(__file__), '..', 'index.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            self.wfile.write(f.read().encode('utf-8'))

    def do_POST(self):
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            pc_key = os.environ.get("PINECONE_API_KEY")
            pc_host = os.environ.get("PINECONE_INDEX_HOST")
            
            client = genai.Client(api_key=api_key)

            content_length = int(self.headers['Content-Length'])
            user_message = json.loads(self.rfile.read(content_length))['message']

            # 1. 修复新版 SDK 语法：降维参数必须放在 config 字典里
            emb_res = client.models.embed_content(
                model="gemini-embedding-001", 
                contents=user_message,
                config={"output_dimensionality": 768}  
            )
            # 删除了多余的重复行
            query_vector = emb_res.embeddings[0].values

            # 查询 Pinecone
            pc_res = requests.post(
                f"{pc_host}/query",
                headers={"Api-Key": pc_key, "Content-Type": "application/json"},
                json={"vector": query_vector, "topK": 2, "includeMetadata": True}
            )
            matches = pc_res.json().get('matches', [])
            context = "\n---\n".join([m['metadata']['text'] for m in matches if 'metadata' in m])

            # 生成回答：已在底层 Prompt 加入纯英文输出设定
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Context:\n{context}\n\nQuestion: {user_message}\n\nAnswer based on context. Note: Ensure all game theory concepts and analysis are purely in English:"
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': response.text}).encode('utf-8'))

        except Exception as e:
            # 2. 核心修复：补全了 HTTP 头部，确保哪怕出错也能把具体的错误文本传回浏览器
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': f"Error: {str(e)}"}).encode('utf-8'))
