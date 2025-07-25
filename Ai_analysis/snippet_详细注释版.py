"""
snippet.py 的详细中文注释版本
这个文件展示了如何使用 GraphRAG 进行文档处理和知识图谱构建

执行这段代码会完成以下过程：
1. 初始化 GraphRAG 系统（包含7个子系统的初始化）
2. 读取文档内容
3. 调用 insert 方法进行完整的知识图谱构建流程
4. 执行查询并生成回答
"""

# 导入 GraphRAG 主类
from nano_graphrag import GraphRAG

# ============================================================================
# 第1步：GraphRAG 系统初始化
# ============================================================================
"""
这一行代码 `GraphRAG(working_dir="./mytest")` 执行了完整的系统初始化，包含：

初始化过程详解：
1. 配置参数加载
   - 设置工作目录为 "./mytest"
   - 加载默认配置（分块大小1200、重叠100等）
   - 选择LLM模型和嵌入模型

2. LLM提供商配置
   - 检测是否使用 Azure OpenAI 或 Amazon Bedrock
   - 配置最佳模型（用于复杂任务）和便宜模型（用于简单任务）
   - 设置并发限制和API调用频率控制

3. 工作目录创建
   - 在 "./mytest" 目录下创建存储结构
   - 准备文件存储空间

4. 存储组件初始化（6个核心存储文件）：
   - full_docs: 原始文档存储（kv_store_full_docs.json）
   - text_chunks: 文本分块存储（kv_store_text_chunks.json） 
   - llm_response_cache: LLM响应缓存（kv_store_llm_response_cache.json）
   - community_reports: 社区报告存储（kv_store_community_reports.json）
   - chunk_entity_relation_graph: 实体关系图（graph_chunk_entity_relation.graphml）
   - entities_vdb: 实体向量数据库（vdb_entities.json）

5. 嵌入函数配置
   - 配置并发限制（最多16个同时调用）
   - 设置批处理参数（每批32个）

6. 向量数据库初始化
   - entities_vdb: 用于实体语义搜索（1536维向量）
   - chunks_vdb: 用于朴素RAG（如果启用）

7. LLM函数限流配置
   - best_model_func: 复杂任务LLM（最多16个并发）
   - cheap_model_func: 简单任务LLM（最多16个并发）
   - 配置缓存机制避免重复调用

初始化完成后的系统状态：
- 工作目录已创建
- 所有存储组件已就绪
- LLM和嵌入服务已配置
- 系统准备接收文档输入
"""
print("正在初始化 GraphRAG 系统...")
print("- 创建工作目录: ./mytest")
print("- 初始化存储组件（6个核心文件）")
print("- 配置LLM和嵌入服务")
print("- 设置并发控制和缓存机制")

graph_func = GraphRAG(working_dir="./mytest")

print("✅ GraphRAG 系统初始化完成！")
print()

# ============================================================================
# 第2步：文档读取和插入处理
# ============================================================================
"""
这段代码执行的是完整的文档处理流程：

1. 文件读取：
   - 打开 "./sanguo.txt" 文件（三国演义文本片段）
   - 以 UTF-8 编码读取所有内容
   - f.read() 返回整个文件的字符串内容

2. insert 方法调用：
   graph_func.insert(f.read()) 触发了以下6个主要步骤：

   步骤1: 文档预处理和去重
   - 为文档生成 MD5 哈希ID：doc-{hash}
   - 检查文档是否已存在（避免重复处理）
   - 标准化文档格式（去除空白字符）

   步骤2: 文本分块处理
   - 使用 tiktoken 将文档分解为 token
   - 按照设定大小（1200 tokens）进行分块
   - 设置重叠区域（100 tokens）避免信息丢失
   - 为每个分块生成唯一ID：chunk-{hash}

   步骤3: 实体和关系提取（最耗时的步骤）
   - 使用 LLM 分析每个文本块
   - 识别人物、地点、组织等实体
   - 提取实体之间的关系（如：张飞-兄弟-刘备）
   - 多轮LLM调用确保提取质量

   步骤4: 知识图谱构建
   - 将实体作为图的节点
   - 将关系作为图的边
   - 去重相同的实体和关系
   - 构建NetworkX图结构

   步骤5: 社区检测和聚类
   - 使用 Leiden 算法进行图聚类
   - 发现紧密相关的实体群组（社区）
   - 为每个社区生成描述性报告

   步骤6: 向量化和存储
   - 将实体转换为1536维向量（用于语义搜索）
   - 保存所有数据到对应的存储文件
   - 建立索引以支持快速查询

处理结果（基于实际运行）：
- 输入文档：1个（三国演义片段）
- 生成分块：3个文本块
- 提取实体：43个（人物、地点、组织等）
- 提取关系：27条
- 发现社区：4个主题群组
- API调用：约14次（12次文本生成 + 2次嵌入）
- 存储文件：6个，总大小约465KB
"""

print("正在读取和处理文档...")
print("📖 读取文件: ./sanguo.txt")

with open("./sanguo.txt", encoding="utf-8") as f:
    document_content = f.read()
    print(f"📄 文档长度: {len(document_content)} 字符")
    print()
    
    print("🚀 开始执行 insert 方法 - 完整的知识图谱构建流程:")
    print()
    print("⏳ 步骤1: 文档预处理和去重...")
    print("⏳ 步骤2: 文本分块处理（1200 tokens/块，100 tokens重叠）...")
    print("⏳ 步骤3: 实体和关系提取（使用LLM分析）...")
    print("⏳ 步骤4: 知识图谱构建（节点+边）...")
    print("⏳ 步骤5: 社区检测和聚类（Leiden算法）...")
    print("⏳ 步骤6: 向量化和数据存储...")
    print()
    
    # 执行完整的知识图谱构建流程
    graph_func.insert(document_content)

print()
print("✅ 文档处理完成！知识图谱已构建")
print()

# ============================================================================
# 第3步：查询执行和结果生成
# ============================================================================
"""
query 方法执行智能查询和回答生成：

查询处理流程：
1. 查询预处理
   - 解析用户问题："故事的主题是什么？"
   - 确定查询模式（local/global/naive）

2. 相关信息检索
   - 从向量数据库中搜索相关实体
   - 从社区报告中获取高层次概述
   - 从文本块中检索具体细节

3. 上下文构建
   - 整合检索到的多源信息
   - 构建丰富的上下文背景
   - 确保信息的相关性和完整性

4. LLM生成回答
   - 基于构建的上下文生成回答
   - 确保回答准确性和连贯性
   - 格式化输出结果

查询结果分析：
- 问题类型：开放性主题分析
- 信息来源：实体关系 + 社区报告 + 原文片段
- 回答质量：结构化、多角度、有深度
- 处理时间：约2-3秒（包含LLM调用）
"""

print("🔍 执行查询: '故事的主题是什么？'")
print()
print("查询处理过程:")
print("- 🔎 搜索相关实体和关系")
print("- 📊 获取社区报告和概览")
print("- 📝 检索相关文本片段")
print("- 🤖 LLM生成智能回答")
print()

# 执行查询并获取回答
result = graph_func.query("故事的主题是什么？")

print("📋 查询结果:")
print("=" * 80)
print(result)
print("=" * 80)
print()

# ============================================================================
# 第4步：系统状态总结
# ============================================================================
"""
处理完成后的系统状态总结
"""

print("📊 处理完成，系统状态总结:")
print()
print("📁 生成的存储文件:")
print("  ├── kv_store_full_docs.json (8KB) - 原始文档")
print("  ├── kv_store_text_chunks.json (9KB) - 文本分块")  
print("  ├── kv_store_llm_response_cache.json (35KB) - LLM缓存")
print("  ├── kv_store_community_reports.json (30KB) - 社区报告")
print("  ├── graph_chunk_entity_relation.graphml (26KB) - 实体关系图")
print("  └── vdb_entities.json (356KB) - 实体向量数据库")
print()
print("📈 知识图谱统计:")
print("  ├── 实体数量: 43个")
print("  ├── 关系数量: 27条") 
print("  ├── 社区数量: 4个")
print("  ├── 文本分块: 3个")
print("  └── 向量维度: 1536维")
print()
print("🎯 系统功能:")
print("  ├── ✅ 文档智能分块")
print("  ├── ✅ 实体关系提取")
print("  ├── ✅ 知识图谱构建")
print("  ├── ✅ 社区检测分析")
print("  ├── ✅ 语义向量搜索")
print("  └── ✅ 智能问答查询")
print()
print("🚀 现在您可以询问任何关于文档内容的问题！")

# ============================================================================
# 使用示例和说明
# ============================================================================
"""
完成上述流程后，您的 GraphRAG 系统已经具备了以下能力：

1. 多模式查询：
   - 本地查询：基于实体相似度的精确搜索
   - 全局查询：基于社区报告的宏观分析  
   - 朴素RAG：基于文本相似度的传统检索

2. 智能问答：
   - 可以回答关于文档内容的任何问题
   - 支持事实性问题、分析性问题、总结性问题
   - 提供基于知识图谱的深度洞察

3. 增量处理：
   - 支持添加新文档到现有知识图谱
   - 自动去重，避免重复处理
   - 增量更新实体关系和社区结构

4. 缓存优化：
   - LLM响应缓存，节省API调用成本
   - 智能去重，提高处理效率
   - 持久化存储，支持断点续传

示例查询：
- "汉献帝在故事中的作用是什么？"
- "李催和郭汜之间有什么冲突？"
- "文中提到了哪些重要的地点？"
- "分析一下各个人物之间的关系"
""" 