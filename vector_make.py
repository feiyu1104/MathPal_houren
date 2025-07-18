import os
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

# 设置 API Key
os.environ['DASHSCOPE_API_KEY'] = 'your-key'  # 替换为你的 API Key

# 调试信息：确认 API Key 是否正确设置
print(f"DASHSCOPE_API_KEY: {os.getenv('DASHSCOPE_API_KEY')}")

# 配置文件路径和向量库存储路径
file_path = "primary_math_docs/grade5.txt"
persist_directory = "vectorstore/grade5"

# 确保存储向量库的目录存在
os.makedirs(persist_directory, exist_ok=True)

# 加载文档
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 提取文档内容
texts = [doc.page_content for doc in docs]

# 切分文档为多个 chunk
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_text(texts[0])  # 假设只有一个文档

# 将 chunks 转换为 Document 对象列表
documents = [Document(page_content=chunk) for chunk in chunks]

# 定义一个简单的嵌入模型类
class DashscopeEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-v4",
                    input=text,
                    dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
                    encoding_format="float"
                )
                # 假设 response.data 是一个包含嵌入向量的列表
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Exception occurred: {e}")
                raise
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-v4",
                input=text,
                dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
                encoding_format="float"
            )
            # 假设 response.data 是一个包含嵌入向量的列表
            return response.data[0].embedding
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise

# 创建嵌入模型实例
api_key = os.getenv('DASHSCOPE_API_KEY')
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
embeddings_model = DashscopeEmbeddings(api_key=api_key, base_url=base_url)

# 存储嵌入向量到向量库
vectorstore = Chroma.from_documents(documents, embeddings_model, persist_directory=persist_directory)
vectorstore.persist()

print(f"向量库已成功保存到 {persist_directory}")