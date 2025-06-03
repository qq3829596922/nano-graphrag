# Nano-GraphRAG 项目结构分析

## 项目概述
**Nano-GraphRAG** 是一个简化、快速、清洁的 GraphRAG 实现，旨在提供易于阅读和修改的代码，同时保持核心功能。整个项目（不包括测试和提示）大约只有 1100 行代码，支持多种存储后端、异步操作且完全类型化。

## 根目录结构

```
nano-graphrag-main/
├── nano_graphrag/              # 主要源代码目录
├── examples/                   # 示例代码和教程
├── tests/                      # 测试代码
├── docs/                       # 文档目录
├── .github/                    # GitHub 配置
├── dickens/                    # 示例工作目录（运行时生成）
├── snippet.py                  # 快速开始示例脚本
├── book.txt                    # 示例文本文件（狄更斯作品）
├── readme.md                   # 项目说明文档
├── setup.py                    # 项目安装配置
├── requirements.txt            # 项目依赖
├── requirements-dev.txt        # 开发依赖
├── LICENSE                     # 许可证文件
├── .gitignore                  # Git 忽略文件
└── MANIFEST.in                 # 打包配置
```

## 核心代码结构 (`nano_graphrag/`)

### 主要模块文件

#### 1. **graphrag.py** (13KB, 368行)
- **作用**: 项目的核心类 `GraphRAG` 定义
- **关键功能**:
  - 主要的 GraphRAG 接口
  - 配置管理和初始化
  - 文档插入和查询的主要逻辑
  - 支持同步和异步操作
  - 多种存储后端的整合

#### 2. **_op.py** (38KB, 1105行) 
- **作用**: 核心操作函数集合
- **关键功能**:
  - 文本分块 (`chunking_by_token_size`)
  - 实体抽取 (`extract_entities`)
  - 社区报告生成 (`generate_community_report`)
  - 本地查询 (`local_query`)
  - 全局查询 (`global_query`)
  - 朴素 RAG 查询 (`naive_query`)

#### 3. **_llm.py** (9.7KB, 295行)
- **作用**: 大语言模型接口
- **支持的模型**:
  - OpenAI GPT-4o/GPT-4o-mini
  - Azure OpenAI
  - Amazon Bedrock
  - 嵌入模型接口

#### 4. **prompt.py** (32KB, 521行)
- **作用**: 提示词模板管理
- **内容**: 包含所有用于实体抽取、关系识别、社区报告生成等任务的提示词模板

#### 5. **_utils.py** (8.8KB, 269行)
- **作用**: 工具函数集合
- **功能**:
  - 哈希计算
  - 异步函数限制
  - JSON 响应转换
  - 事件循环管理
  - 日志记录

#### 6. **_splitter.py** (3.5KB, 95行)
- **作用**: 文本分割器
- **功能**: 提供多种文本分割策略

#### 7. **base.py** (5.4KB, 187行)
- **作用**: 基类定义
- **内容**:
  - `BaseKVStorage`: 键值存储基类
  - `BaseVectorStorage`: 向量存储基类  
  - `BaseGraphStorage`: 图存储基类
  - `QueryParam`: 查询参数类

### 存储模块 (`_storage/`)

#### 键值存储
- **kv_json.py**: JSON 文件存储实现

#### 向量数据库
- **vdb_nanovectordb.py**: 内置的 nano-vectordb 实现
- **vdb_hnswlib.py**: HNSW 算法向量数据库

#### 图数据库
- **gdb_networkx.py**: NetworkX 图存储实现
- **gdb_neo4j.py**: Neo4j 图数据库实现

### 实体抽取模块 (`entity_extraction/`)

- **extract.py**: 实体抽取核心逻辑
- **module.py**: 实体抽取模块定义
- **metric.py**: 评估指标

## 示例代码 (`examples/`)

### 核心示例
- **no_openai_key_at_all.py**: 不使用 OpenAI API 的完整示例
- **using_ollama_as_llm.py**: 使用 Ollama 作为 LLM
- **using_deepseek_as_llm.py**: 使用 DeepSeek API

### 向量数据库示例
- **using_faiss_as_vextorDB.py**: 使用 FAISS
- **using_milvus_as_vectorDB.py**: 使用 Milvus
- **using_qdrant_as_vectorDB.py**: 使用 Qdrant
- **using_hnsw_as_vectorDB.py**: 使用 HNSW

### 其他集成示例
- **using_amazon_bedrock.py**: 使用 Amazon Bedrock
- **using_custom_chunking_method.py**: 自定义分块方法
- **using_local_embedding_model.py**: 使用本地嵌入模型

### 可视化和分析
- **graphml_visualize.py**: GraphML 可视化
- **benchmarks/**: 性能基准测试

## 测试代码 (`tests/`)

### 测试文件
- **test_rag.py**: RAG 功能测试
- **test_openai.py**: OpenAI API 测试
- **test_splitter.py**: 文本分割器测试
- **test_json_parsing.py**: JSON 解析测试
- **test_networkx_storage.py**: NetworkX 存储测试
- **test_neo4j_storage.py**: Neo4j 存储测试
- **test_hnsw_vector_storage.py**: HNSW 向量存储测试

### 测试数据
- **mock_data.txt**: 测试用的狄更斯作品
- **zhuyuanzhang.txt**: 中文测试数据
- **fixtures/**: 测试夹具

## 文档 (`docs/`)

- **benchmark-en.md**: 英文基准测试
- **benchmark-zh.md**: 中文基准测试  
- **benchmark-dspy-entity-extraction.md**: DSPy 实体抽取基准
- **FAQ.md**: 常见问题
- **ROADMAP.md**: 路线图
- **CONTRIBUTING.md**: 贡献指南
- **use_neo4j_for_graphrag.md**: Neo4j 使用指南

## 架构设计特点

### 1. **模块化设计**
- 核心功能分离到不同模块
- 存储后端可插拔
- LLM 和嵌入模型可替换

### 2. **异步支持** 
- 所有核心操作都支持异步执行
- 使用 `asyncio` 进行并发处理
- 异步函数调用限制

### 3. **可扩展性**
- 支持多种向量数据库（FAISS, Milvus, Qdrant 等）
- 支持多种 LLM（OpenAI, Azure, Bedrock, Ollama 等）
- 支持多种图存储（NetworkX, Neo4j）

### 4. **配置驱动**
- 使用 `dataclass` 进行配置管理
- 支持运行时配置修改
- 详细的配置参数文档

### 5. **缓存机制**
- LLM 响应缓存
- 向量嵌入缓存
- 增量更新支持

## 工作流程

### 插入流程
1. **文档处理**: 计算文档哈希，避免重复
2. **文本分块**: 使用 token 大小进行分块
3. **实体抽取**: 从文本块中抽取实体和关系
4. **图构建**: 构建知识图谱
5. **社区检测**: 对图进行聚类
6. **报告生成**: 为每个社区生成摘要报告
7. **向量化**: 对实体和文本进行向量嵌入
8. **存储**: 将所有数据持久化存储

### 查询流程
- **Local 模式**: 基于实体向量相似性进行局部查询
- **Global 模式**: 基于社区报告进行全局查询  
- **Naive 模式**: 传统 RAG 向量相似性查询

## 依赖管理

### 核心依赖 (requirements.txt)
- `tiktoken`: Token 计算
- `dataclasses-json`: 数据类 JSON 序列化
- `tenacity`: 重试机制
- `nano-vectordb`: 内置向量数据库
- `networkx`: 图处理
- `numpy`: 数值计算

### 开发依赖 (requirements-dev.txt)
- `pytest`: 测试框架
- `coverage`: 代码覆盖率
- `black`: 代码格式化

## 使用建议

### 1. **快速开始**
- 使用 `snippet.py` 作为起点
- 设置 OpenAI API 密钥
- 使用默认配置即可开始

### 2. **生产环境**
- 选择合适的向量数据库（如 FAISS 或 Milvus）
- 配置缓存策略
- 调整分块大小和重叠参数

### 3. **自定义扩展**
- 查看 `examples/` 目录获取集成示例
- 实现自定义存储后端
- 添加新的 LLM 支持

### 4. **性能优化**
- 调整异步并发参数
- 使用合适的图聚类算法
- 优化嵌入批处理大小

这个项目的设计非常优雅，代码结构清晰，模块化程度高，是学习和实现 GraphRAG 的优秀参考。 