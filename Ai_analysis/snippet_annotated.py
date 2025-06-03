# 导入必要的模块
from nano_graphrag import GraphRAG, QueryParam  # 导入GraphRAG主类和查询参数类
from dotenv import load_dotenv  # 导入环境变量加载器

# 加载环境变量文件（.env），override=True表示覆盖已存在的环境变量
# 这通常用于加载OpenAI API密钥等敏感信息
load_dotenv(override=True)

# ===== GraphRAG 初始化 =====
# 这行代码是整个程序的核心，创建一个GraphRAG实例
# working_dir="./mytest" 指定工作目录为当前目录下的mytest文件夹
# 
# 初始化过程中会发生以下操作：
# 1. 创建工作目录 ./mytest（如果不存在）
# 2. 初始化多个存储组件：
#    - kv_store_full_docs.json: 存储完整文档
#    - kv_store_text_chunks.json: 存储文本分块
#    - kv_store_llm_response_cache.json: 缓存LLM响应
#    - kv_store_community_reports.json: 存储社区报告
#    - graph_chunk_entity_relation.graphml: 存储实体关系图
#    - vdb_entities.json: 存储实体向量数据库
# 3. 设置LLM函数的并发限制和缓存机制
# 4. 配置文本嵌入函数和向量数据库
graph_func = GraphRAG(working_dir="./mytest")

# ===== 文档插入 =====
# 读取三国演义文本文件并插入到GraphRAG系统中
# 这个过程会：
# 1. 将文档内容分块（默认每块1200个token，重叠100个token）
# 2. 使用LLM提取实体和关系
# 3. 构建知识图谱
# 4. 进行图聚类，生成社区
# 5. 为每个社区生成摘要报告
# 6. 将实体向量化并存储到向量数据库
with open("./sanguo.txt", encoding="utf-8") as f:
    graph_func.insert(f.read())

# ===== 查询操作 =====

# 全局查询（默认模式）
# 使用社区报告进行查询，适合回答需要全局理解的问题
# 查询过程：
# 1. 分析查询意图
# 2. 检索相关的社区报告
# 3. 基于社区报告生成答案
print("=== 全局查询结果 ===")
print(graph_func.query("故事的主题是什么，用中文回答"))

# 本地查询模式
# 使用实体向量数据库进行查询，适合回答具体的事实性问题
# 查询过程：
# 1. 将查询转换为向量
# 2. 在实体向量数据库中搜索相似实体
# 3. 检索相关的文本块
# 4. 基于检索到的信息生成答案
print("\n=== 本地查询结果 ===")
print(graph_func.query("故事的主题是什么，用中文回答?", param=QueryParam(mode="local")))

# 再次进行全局查询（用于对比）
# 注意：这里的查询内容与第一次略有不同（多了问号）
print("\n=== 全局查询结果（第二次） ===")
print(graph_func.query("故事的主题是什么，用中文回答?", param=QueryParam(mode="global")))

# ===== 查询模式说明 =====
# 
# 1. global模式（默认）：
#    - 使用社区报告进行查询
#    - 适合需要全局理解的问题
#    - 查询速度较快，但可能缺少细节
#
# 2. local模式：
#    - 使用实体向量数据库进行查询
#    - 适合具体的事实性问题
#    - 查询结果更详细，但速度较慢
#
# 3. naive模式（未启用）：
#    - 简单的向量相似度查询
#    - 类似传统的RAG方法
#    - 需要设置 enable_naive_rag=True 才能使用 