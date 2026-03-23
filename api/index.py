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

            # 向量化问题（必须保留 config 语法以适配最新 SDK）
            emb_res = client.models.embed_content(
                model="gemini-embedding-001", 
                contents=user_message,
                config={"output_dimensionality": 768}
            )
            query_vector = emb_res.embeddings[0].values

            # 查询 Pinecone 
            pc_res = requests.post(
                f"{pc_host}/query",
                headers={"Api-Key": pc_key, "Content-Type": "application/json"},
                json={"vector": query_vector, "topK": 2, "includeMetadata": True}
            )
            matches = pc_res.json().get('matches', [])
            context = "\n---\n".join([m['metadata']['text'] for m in matches if 'metadata' in m])

            # --- 核心修改点：切换模型为 gemini-1.5-flash ---
            response = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=f"Context:\n{context}\n\nQuestion: {user_message}\n\nAnswer based on context. Note: Ensure all game theory concepts and analysis are purely in English:"
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': response.text}).encode('utf-8'))

        except Exception as e:
            # 调试日志：如果依然报错，Vercel Logs 会显示具体原因
            print(f"🔥 FATAL ERROR CAUGHT: {str(e)}")
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': f"Error: {str(e)}"}).encode('utf-8'))
