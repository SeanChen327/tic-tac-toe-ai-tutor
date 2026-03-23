from http.server import BaseHTTPRequestHandler
import json
import os

# 仅引入 LangChain 核心与 Google 模块，保持极度轻量
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

vector_store = None

def get_vector_store(api_key):
    global vector_store
    if vector_store is not None:
        return vector_store

    # 1. 使用纯 Python 读取文件，省去加载沉重的 langchain-community
    file_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    # 将纯文本包装成 LangChain 需要的 Document 格式
    docs = [Document(page_content=text_content)]

    # 2. 文本切块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. 初始化 Embedding 模型
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )

    # 4. 使用纯 Python 的内存向量库替代 FAISS！完美兼容 Vercel
    vector_store = InMemoryVectorStore.from_documents(splits, embeddings)
    return vector_store

class handler(BaseHTTPRequestHandler):
    
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
            
            # 获取极速版内存向量库
            v_store = get_vector_store(api_key)
            retriever = v_store.as_retriever(search_kwargs={"k": 2})

            # 初始化大模型
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.3
            )

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

            # 执行 RAG 检索链
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
