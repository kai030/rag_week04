# 加载文档
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
#创建文档摘要 (使用原始文档内容作为摘要以避免API调用)
summaries = [doc.page_content for doc in docs]
# 设置多向量检索器
from langchain_core.stores import InMemoryByteStore # 内存存储
from langchain_huggingface import HuggingFaceEmbeddings # 向量模型
from langchain_community.vectorstores import Chroma # 向量数据库
from langchain_experimental.retrievers.multi_vector import MultiVectorRetriever # 多向量检索器
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh") # 向量模型
vectorstore = Chroma(collection_name="summaries", embedding_function= embed_model) # 向量数据库
store = InMemoryByteStore() # 内存存储
id_key = "doc_id" # 文档ID
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
# 添加文档和摘要到检索器
import uuid
from langchain_core.documents import Document
doc_ids = [str(uuid.uuid4()) for _ in docs]
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
# 使用检索器进行查询
query = "Memory in agents"
retrieved_docs = retriever.get_relevant_documents(query,n_results=1)

print(retrieved_docs)
