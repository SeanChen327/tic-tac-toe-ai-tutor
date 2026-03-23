from http.server import BaseHTTPRequestHandler
import json
import os
from google import genai
from google.genai import types

# 引入 LangChain 核心组件
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 全局变量缓存向量数据库，防止 Vercel 每次请求都重新读取文件
vector_store = None

def get_vector_store(api_key):
    global vector_store
    if vector_store is not None:
        return vector_store

    # 1. 加载我们在根目录准备好的知识库
    file_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge.txt')
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # 2. 将长文本切分成小块 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. 初始化 Google 的向量化模型 (Embedding)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )

    # 4. 使用 FAISS 构建本地内存级别的向量数据库
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

class handler(BaseHTTPRequestHandler):
    
    # 保持原样的 GET 方法，用于展示游戏界面
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        try:
            html_path = os.path.join(os.path.dirname(__file__), '..', 'index.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            self.wfile.write(html_content.encode('utf-8'))
        except Exception as e:
            error_page = f"<h1>Loading Error</h1><p>{str(e)}</p>"
            self.wfile.write(error_page.encode('utf-8'))

    # 重构后的 POST 方法，接入 LangChain RAG 架构
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
                raise Exception("API Key missing.")
            
            # 获取或初始化向量数据库
            v_store = get_vector_store(api_key)
            retriever = v_store.as_retriever(search_kwargs={"k": 2}) # 每次检索最相关的 2 个文本块

            # 初始化 LLM 模型
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.3 # 降低随机性，让回答更严谨
            )

            # 设定 RAG 的系统提示词模板
            system_prompt = (
                "You are an expert Tic-Tac-Toe AI assistant. "
                "Use the following pieces of retrieved game theory context to answer the player's question. "
                "If you don't know the answer based on the context, just say that you don't know. "
                "Always respond in English concisely and professionally.\n\n"
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # 将检索器和语言模型链式组合
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # 执行查询
            response = rag_chain.invoke({"input": user_message})
            ai_reply = response["answer"]

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'reply': ai_reply}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_msg = f"RAG Backend Error: {str(e)}"
            self.wfile.write(json.dumps({'reply': error_msg}).encode('utf-8'))
