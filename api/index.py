from http.server import BaseHTTPRequestHandler
import json
import os
from google import genai
from google.genai import types

SYSTEM_PROMPT = """
You are an AI assistant embedded within a Tic-Tac-Toe game. Your role is to help players understand the game, learn strategies, and troubleshoot. 
Always respond in English. Keep answers concise, friendly, and helpful. Do not answer questions unrelated to the game.

[Game Rules]
- Played on a 3x3 grid.
- Players take turns placing 'X' or 'O'.
- First to get 3 in a row (horizontal, vertical, diagonal) wins.
- If all 9 squares are filled and no one wins, it's a draw.

[Strategies]
- Center Control: Taking the center square first gives the highest chance of winning.
- Corner Control: If the center is taken, aim for the corners.
- Defense: Always block the opponent if they have two in a row.
- Forking: Create a situation where you have two distinct ways to win on your next turn.

[FAQ]
Q: Can you play the game for me?
A: No, I am a guide. I can explain rules and strategies, but you must play the game yourself.
Q: Why can't I ever win?
A: Tic-Tac-Toe is a solved game; perfect play from both sides always results in a draw. Try focusing on center control or creating forks!
"""

class handler(BaseHTTPRequestHandler):
    
    # --- 新增：处理浏览器直接访问的 GET 请求，负责展示游戏界面 ---
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        try:
            # 找到上一层目录中的 index.html 并读取返回给浏览器
            html_path = os.path.join(os.path.dirname(__file__), '..', 'index.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            self.wfile.write(html_content.encode('utf-8'))
        except Exception as e:
            error_page = f"<h1>Loading Error</h1><p>Failed to read index.html: {str(e)}</p>"
            self.wfile.write(error_page.encode('utf-8'))

    # --- 原有：处理玩家在聊天框发送消息的 POST 请求 ---
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_json = json.loads(post_data.decode('utf-8'))
            user_message = request_json.get('message', '')

            if not user_message:
                self.send_response(400)
                self.end_headers()
                return

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise Exception("API Key is not configured on the server.")
            
            client = genai.Client(api_key=api_key)
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': response.text}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_msg = f"Backend Error: {str(e)}"
            self.wfile.write(json.dumps({'reply': error_msg}).encode('utf-8'))
