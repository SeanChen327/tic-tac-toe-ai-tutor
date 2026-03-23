from http.server import BaseHTTPRequestHandler
import json
import os
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
            # 现在只需要 Gemini 的 API Key 即可
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)

            content_length = int(self.headers['Content-Length'])
            user_message = json.loads(self.rfile.read(content_length))['message']

            # 核心改造：将 Pinecone 的知识库直接硬编码到 Prompt 中
            KNOWLEDGE_BASE = """
            Tic-Tac-Toe Game Theory & Strategies Context:
            1. Basic Rules: Two players (X and O) take turns marking spaces in a 3x3 grid. The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row wins the game.
            2. Win: If the player has two in a row, they can place a third to get three in a row.
            3. Block: If the opponent has two in a row, the player must play the third themselves to block the opponent.
            4. Fork (叉子战术): Create an opportunity where the player has two ways to win (two non-blocked lines of 2).
            5. Blocking an opponent's fork: If there is only one possible fork for the opponent, the player should block it. Otherwise, the player should create a two-in-a-row to force the opponent into defending.
            6. Center (占据中心): A player marks the center.
            7. Opposite corner (对角线陷阱): If the opponent is in the corner, the player plays the opposite corner.
            8. Empty corner: The player plays in a corner square.
            9. Empty side: The player plays in a middle square on any of the 4 sides.
            Note: Optimal play from both sides always results in a draw.
            """

            # 组合 Prompt 并进行单次 API 调用
            prompt = f"System Context:\n{KNOWLEDGE_BASE}\n\nUser Question: {user_message}\n\nTask: Answer the user's question based on the provided context. Note: Ensure all game theory concepts and analysis are purely in English."

            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': response.text}).encode('utf-8'))

        except Exception as e:
            # 依然保留兜底机制，方便在 Vercel 日志中排错
            print(f"🔥 FATAL ERROR CAUGHT: {str(e)}")
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': f"Error: {str(e)}"}).encode('utf-8'))
