# 导入必要的标准库和第三方库
import asyncio  # 异步编程支持
import os  # 操作系统接口
from dataclasses import asdict, dataclass, field  # 数据类装饰器和工具
from datetime import datetime  # 日期时间处理
from functools import partial  # 函数式编程工具
from typing import Callable, Dict, List, Optional, Type, Union, cast  # 类型注解

import tiktoken  # OpenAI的分词器，用于计算token数量


from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    get_chunks,
    local_query,
    global_query,
    naive_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


@dataclass
class GraphRAG:
    """
    GraphRAG核心类，用于构建和查询基于图的检索增强生成系统
    
    示例：
    rag = GraphRAG(working_dir="./my_rag_cache")
    rag.insert(["文档1内容", "文档2内容"])
    result = rag.query("查询问题")
    """
    
    # 工作目录设置
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().stYrftime('%-%m-%d-%H:%M:%S')}"
    )
    # 工作目录路径，用于存储所有缓存文件和数据
    # 示例："./nano_graphrag_cache_2024-01-15-14:30:45"
    
    # 图模式开关
    enable_local: bool = True
    # 是否启用本地查询模式，基于实体和社区进行查询
    # 示例：True表示可以使用param.mode="local"进行查询
    
    enable_naive_rag: bool = True
    # 是否启用简单RAG模式，基于文本块相似度进行查询
    # 示例：True表示可以使用param.mode="naive"进行查询

    # 文本分块配置
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    # 文本分块函数，将长文档分割成小块
    # 示例：chunking_by_token_size函数将文档按token数量分块
    
    chunk_token_size: int = 1200
    # 每个文本块的最大token数量
    # 示例：1200表示每个块最多包含1200个token（约800-1000个中文字符）
    
    chunk_overlap_token_size: int = 100
    # 相邻文本块之间的重叠token数量，用于保持上下文连续性
    # 示例：100表示相邻块之间重叠100个token
    
    tiktoken_model_name: str = "gpt-4o"
    # 用于token计算的模型名称
    # 示例："gpt-4o"表示使用GPT-4o的分词器计算token

    # 实体提取配置
    entity_extract_max_gleaning: int = 1
    # 实体提取的最大迭代次数，用于提高提取质量
    # 示例：1表示对每个文本块进行1次实体提取迭代
    
    entity_summary_to_max_tokens: int = 500
    # 实体摘要的最大token数量
    # 示例：500表示每个实体的描述摘要最多500个token
    
    # 图聚类配置
    graph_cluster_algorithm: str = "leiden"
    # 图聚类算法，用于将相关实体分组成社区
    # 示例："leiden"表示使用Leiden算法进行社区检测
    
    max_graph_cluster_size: int = 10
    # 每个图聚类的最大实体数量
    # 示例：10表示每个社区最多包含10个实体
    
    graph_cluster_seed: int = 0xDEADBEEF
    # 图聚类的随机种子，确保结果可重现
    # 示例：0xDEADBEEF（十六进制）用于随机数生成器初始化

    # 节点嵌入配置
    node_embedding_algorithm: str = "node2vec"
    # 节点嵌入算法，用于将图中的节点转换为向量表示
    # 示例："node2vec"表示使用Node2Vec算法生成节点嵌入
    
    # node2vec_params: dict = field(
    #     default_factory=lambda: {
    #         "dimensions": 1536,
    #         "num_walks": 10,
    #         "walk_length": 40,
    #         "num_walks": 10,
    #         "window_size": 2,
    #         "iterations": 3,
    #         "random_seed": 3,
    #     }
    # )
    # Node2Vec算法的参数配置
    # 示例：{
    #   "dimensions": 1536,    # 嵌入向量维度
    #   "num_walks": 10,       # 每个节点的随机游走次数
    #   "walk_length": 40,     # 每次随机游走的长度
    #   "window_size": 2,      # 上下文窗口大小
    #   "iterations": 3,       # 训练迭代次数
    #   "random_seed": 3       # 随机种子
    # }
    
    # 社区报告配置
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    # 生成社区报告时的LLM特殊参数
    # 示例：{"response_format": {"type": "json_object"}} 表示要求LLM返回JSON格式的响应

    # 文本嵌入配置
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    # 文本嵌入函数，用于将文本转换为向量表示
    # 示例：openai_embedding函数将文本转换为1536维的向量
    
    embedding_batch_num: int = 32
    # 每批处理的嵌入数量，用于批量处理提高效率
    # 示例：32表示每次并行处理32个文本的嵌入计算
    
    embedding_func_max_async: int = 16
    # 嵌入函数的最大并发数量
    # 示例：16表示最多同时执行16个嵌入计算任务
    
    query_better_than_threshold: float = 0.2
    # 查询相似度阈值，用于过滤不相关的结果
    # 示例：0.2表示只返回相似度大于0.2的查询结果

    # LLM配置
    using_azure_openai: bool = False
    # 是否使用Azure OpenAI服务
    # 示例：True表示使用Azure OpenAI而不是标准OpenAI API
    
    using_amazon_bedrock: bool = False
    # 是否使用Amazon Bedrock服务
    # 示例：True表示使用AWS Bedrock上的LLM模型
    
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    # 最佳模型ID，用于复杂任务（如实体提取、社区报告生成）
    # 示例："us.anthropic.claude-3-sonnet-20240229-v1:0"是Claude-3 Sonnet模型
    
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    # 便宜模型ID，用于简单任务以节省成本
    # 示例："us.anthropic.claude-3-haiku-20240307-v1:0"是Claude-3 Haiku模型
    
    best_model_func: callable = gpt_4o_complete
    # 最佳模型的调用函数
    # 示例：gpt_4o_complete函数用于调用GPT-4o模型
    
    best_model_max_token_size: int = 32768
    # 最佳模型的最大token长度限制
    # 示例：32768表示模型最多处理32K个token的上下文
    
    best_model_max_async: int = 16
    # 最佳模型的最大并发调用数
    # 示例：16表示最多同时调用16个模型实例
    
    cheap_model_func: callable = gpt_4o_mini_complete
    # 便宜模型的调用函数
    # 示例：gpt_4o_mini_complete函数用于调用GPT-4o-mini模型
    
    cheap_model_max_token_size: int = 32768
    # 便宜模型的最大token长度限制
    # 示例：32768表示模型最多处理32K个token的上下文
    
    cheap_model_max_async: int = 16
    # 便宜模型的最大并发调用数
    # 示例：16表示最多同时调用16个模型实例

    # 实体提取函数配置
    entity_extraction_func: callable = extract_entities
    # 实体提取函数，用于从文本中提取实体和关系
    # 示例：extract_entities函数从文本块中识别人名、地名、组织等实体及其关系
    
    # 存储配置
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # 键值对存储类，用于存储文档、文本块等数据
    # 示例：JsonKVStorage将数据以JSON格式存储在文件系统中
    
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    

    # 示例：NanoVectorDBStorage提供向量相似度搜索功能
    
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    # 向量数据库存储类的额外参数
    # 示例：{"index_type": "IVF", "metric": "cosine"} 指定索引类型和距离度量
    
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    # 图存储类，用于存储实体关系图
    # 示例：NetworkXStorage使用NetworkX库存储和操作图结构
    
    enable_llm_cache: bool = True
    # 是否启用LLM响应缓存，避免重复调用相同的LLM请求
    # 示例：True表示缓存LLM响应以提高性能和节省成本

    # 扩展配置
    always_create_working_dir: bool = True
    # 是否总是创建工作目录，即使目录不存在
    # 示例：True表示自动创建不存在的工作目录
    
    addon_params: dict = field(default_factory=dict)
    # 附加参数字典，用于存储自定义配置
    # 示例：{"custom_param": "value", "debug_mode": True} 存储用户自定义参数
    
    convert_response_to_json_func: callable = convert_response_to_json
    # 将LLM响应转换为JSON格式的函数
    # 示例：convert_response_to_json函数解析LLM的文本响应并转换为Python字典

    def __post_init__(self):
        """
        数据类初始化后的自动配置方法
        
        过程示例：
        1. 打印所有配置参数到日志
        2. 根据使用的云服务切换LLM和嵌入函数
        3. 创建工作目录
        4. 初始化各种存储实例
        """
        # 打印所有配置参数用于调试
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        # 如果使用Azure OpenAI，切换到对应的Azure函数
        if self.using_azure_openai:
            # 将默认的OpenAI函数替换为Azure OpenAI对应函数
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
                # 示例：将gpt_4o_complete替换为azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
                # 示例：将gpt_4o_mini_complete替换为azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
                # 示例：将openai_embedding替换为azure_openai_embedding
            logger.info(
                "已将默认的OpenAI函数切换为Azure OpenAI对应函数"
            )

        # 如果使用Amazon Bedrock，切换到对应的Bedrock函数
        if self.using_amazon_bedrock:
            # 创建Amazon Bedrock的模型调用函数
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            # 示例：为Claude-3 Sonnet模型创建Bedrock调用函数
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            # 示例：为Claude-3 Haiku模型创建Bedrock调用函数
            self.embedding_func = amazon_bedrock_embedding
            # 示例：使用Amazon Bedrock的嵌入服务
            logger.info(
                "已将默认的OpenAI函数切换为Amazon Bedrock对应函数"
            )

        # 创建工作目录（如果不存在且允许创建）
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"正在创建工作目录 {self.working_dir}")
            os.makedirs(self.working_dir)
            # 示例：创建"./nano_graphrag_cache_2024-01-15-14:30:45"目录

        # 初始化各种存储实例
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        # 完整文档存储，保存原始文档内容
        # 示例：存储{"doc-123": {"content": "这是一篇文档..."}}

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        # 文本块存储，保存分块后的文档片段
        # 示例：存储{"chunk-456": {"content": "文档片段...", "doc_id": "doc-123"}}

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        # LLM响应缓存，避免重复调用相同的LLM请求
        # 示例：存储{"query_hash": {"response": "LLM的回答..."}}

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        # 社区报告存储，保存图聚类的社区摘要
        # 示例：存储{"community-1": {"summary": "这个社区包含了关于AI的实体..."}}
        
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )
        # 块-实体关系图存储，保存实体和关系的图结构
        # 示例：存储实体节点和它们之间的关系边

        # 初始化嵌入函数并限制并发数
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # 示例：将嵌入函数包装为最多16个并发调用的限制版本
        
        # 初始化实体向量数据库（仅在启用本地查询时）
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        # 实体向量数据库，存储实体的向量表示用于相似度搜索
        # 示例：存储{"entity_id": vector, "entity_name": "苹果公司"}
        
        # 初始化文本块向量数据库（仅在启用简单RAG时）
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        # 文本块向量数据库，存储文本块的向量表示用于检索
        # 示例：存储{"chunk_id": vector, "content": "文档内容片段..."}

        # 初始化LLM函数并限制并发数，同时添加缓存支持
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        # 最佳模型函数，用于复杂任务，带有缓存和并发限制
        # 示例：最多16个并发调用，自动缓存相同请求的响应
        
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )
        # 便宜模型函数，用于简单任务，带有缓存和并发限制
        # 示例：最多16个并发调用，自动缓存相同请求的响应

    def insert(self, string_or_strings):
        """
        同步插入文档的方法
        
        过程示例：
        1. 调用者：rag.insert(["文档1", "文档2"])
        2. 获取事件循环并执行异步插入
        3. 返回插入结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        同步查询方法
        
        过程示例：
        1. 调用者：result = rag.query("什么是人工智能？", QueryParam(mode="local"))
        2. 获取事件循环并执行异步查询
        3. 返回查询结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步查询方法，根据不同模式执行不同的查询策略
        
        过程示例：
        1. 输入："什么是人工智能？", QueryParam(mode="local", top_k=5)
        2. 检查模式是否启用
        3. 根据模式选择查询策略：
           - local: 基于实体和社区的本地查询
           - global: 基于整个知识图谱的全局查询  
           - naive: 基于文本块相似度的简单RAG查询
        4. 执行查询并返回结果
        5. 清理查询缓存
        """
        # 检查查询模式是否启用
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local为False，无法使用local模式查询")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag为False，无法使用naive模式查询")
            
        # 根据不同模式执行相应的查询策略
        if param.mode == "local":
            # 本地查询：基于相关实体和社区进行查询
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
            # 示例：查找与"人工智能"相关的实体和社区，生成基于局部知识的回答
            
        elif param.mode == "global":
            # 全局查询：基于整个知识图谱进行查询
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
            # 示例：综合所有社区报告，生成基于全局知识的回答
            
        elif param.mode == "naive":
            # 简单RAG查询：基于文本块相似度进行查询
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
            # 示例：找到与查询最相似的文本块，基于这些块生成回答
            
        else:
            raise ValueError(f"未知查询模式 {param.mode}")
            
        # 执行查询完成后的清理工作
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        """
        异步插入文档的核心方法
        过程示例：
        1. 输入：["文档1内容", "文档2内容"]
        2. 开始插入操作的准备工作
        3. 处理新文档（去重、计算哈希ID）
        4. 文本分块处理
        5. 实体提取和关系构建
        6. 图聚类和社区报告生成
        7. 提交所有数据到存储
        8. 完成插入操作的清理工作
        """
        # 开始插入操作，初始化各种存储的索引回调
        await self._insert_start()
        try:
            # 确保输入是列表格式
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
                
            # ---------- 处理新文档
            # 为每个文档生成唯一的哈希ID并创建文档对象
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            # 示例：{"doc-abc123": {"content": "这是一篇关于AI的文档..."}}
            
            # 过滤出不在存储中的新文档，避免重复插入
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))

            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            
            # 检查是否有新文档需要处理
            if not len(new_docs):
                logger.warning(f"所有文档都已存在于存储中")
                return
            logger.info(f"[新文档] 正在插入 {len(new_docs)} 个文档")

            # ---------- 文本分块处理

            # 获取新文档的文本块，使用分块函数将文档分割成合适大小的块
            # inserting_chunks = get_chunks(
            #     new_docs=new_docs,  # 新文档数据
            #     chunk_func=self.chunk_func,  # 分块函数
            #     overlap_token_size=self.chunk_overlap_token_size,  # 文本块重叠的token数量
            #     max_token_size=self.chunk_token_size,  # 每个文本块的最大token数量
            # )
            # # 示例输出：{"chunk-def456": {"content": "AI是一种...", "doc_id": "doc-abc123", "chunk_index": 0}}

            # # 从文本块存储中过滤出需要添加的块的键，避免重复插入
            # _add_chunk_keys = await self.text_chunks.filter_keys(
            #     list(inserting_chunks.keys())  # 获取所有插入块的键列表
            # )

            # # 过滤插入的文本块，只保留那些不在存储中的新块
            # inserting_chunks = {
            #     k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            # }

            # # 检查是否有新的文本块需要插入
            # if not len(inserting_chunks):
            #     logger.warning(f"所有文本块都已存在于存储中")  # 所有块都已存在于存储中的警告
            #     return  # 如果没有新块，直接返回
            # logger.info(f"[新文本块] 正在插入 {len(inserting_chunks)} 个文本块")  # 记录插入的新块数量
            
            # # 如果启用了简单RAG模式，将文本块插入到向量数据库中
            # if self.enable_naive_rag:
            #     logger.info("为简单RAG模式插入文本块")  # 记录为简单RAG插入块的日志
            #     await self.chunks_vdb.upsert(inserting_chunks)  # 将文本块插入到向量数据库
            #     # 示例：将文本块转换为向量并存储，支持后续的相似度搜索

            # # TODO: 目前社区报告没有增量更新功能，所以删除所有现有的社区报告
            # await self.community_reports.drop()
            # # 注意：这是一个临时解决方案，会删除所有现有的社区报告

            # # ---------- 实体提取和关系构建
            # logger.info("[实体提取]...")  # 记录开始实体提取的日志
            # # 执行实体提取函数，从文本块中提取实体和关系
            # maybe_new_kg = await self.entity_extraction_func(
            #     inserting_chunks,  # 要处理的文本块
            #     knwoledge_graph_inst=self.chunk_entity_relation_graph,  # 知识图谱实例
            #     entity_vdb=self.entities_vdb,  # 实体向量数据库
            #     global_config=asdict(self),  # 全局配置参数
            #     using_amazon_bedrock=self.using_amazon_bedrock,  # 是否使用Amazon Bedrock
            # )
            # # 示例过程：
            # # 1. LLM分析文本块："苹果公司发布了新的iPhone"
            # # 2. 提取实体：["苹果公司", "iPhone"] 
            # # 3. 识别关系：("苹果公司", "发布", "iPhone")
            # # 4. 更新知识图谱和向量数据库
            
            # # 检查是否找到了新的实体
            # if maybe_new_kg is None:
            #     logger.warning("未找到新实体")  # 没有找到新实体的警告
            #     return  # 如果没有新实体，直接返回
            # self.chunk_entity_relation_graph = maybe_new_kg  # 更新块-实体关系图
            
            # # ---------- 图聚类和社区分析
            # logger.info("[社区报告]...")  # 记录开始社区报告生成的日志
            # # 对知识图谱进行聚类分析，识别实体社区
            # await self.chunk_entity_relation_graph.clustering(
            #     self.graph_cluster_algorithm  # 图聚类算法
            # )
            # # 示例过程：
            # # 1. 使用Leiden算法对实体图进行聚类
            # # 2. 将相关实体分组成社区，如：["苹果公司", "iPhone", "iOS"] 组成科技产品社区
            
            # # 生成社区报告，总结每个社区的信息
            # await generate_community_report(
            #     self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            # )
            # 示例过程：
            # 1. 分析每个社区的实体和关系
            # 2. 使用LLM生成社区摘要："这个社区主要涉及苹果公司的产品和技术..."
            # 3. 存储社区报告用于后续的全局查询

            # ---------- 提交所有数据到存储
            await self.full_docs.upsert(new_docs)  # 将新文档插入到完整文档存储中
            # await self.text_chunks.upsert(inserting_chunks)  # 将新文本块插入到文本块存储中
            # 示例：持久化所有处理后的数据，确保数据不会丢失
            
        finally:
            await self._insert_done()  # 执行插入完成后的清理操作
            # 示例：关闭数据库连接、保存索引文件、清理临时数据等

    async def _insert_start(self):
        """
        插入操作开始时的准备工作
        
        过程示例：
        1. 准备图存储的索引回调
        2. 并行执行所有存储实例的开始回调
        3. 为批量插入操作做准备
        """
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)



    async def _insert_done(self):
        """
        插入操作完成后的清理工作
        
        过程示例：
        1. 并行执行所有存储实例的完成回调
        2. 保存索引文件、提交事务、关闭连接等
        3. 确保所有数据都已持久化
        """
        tasks = []
        for storage_inst in [
            self.full_docs,             # 完整文档存储
            self.text_chunks,           # 文本块存储
            self.llm_response_cache,    # LLM响应缓存
            self.community_reports,     # 社区报告存储
            self.entities_vdb,          # 实体向量数据库
            self.chunks_vdb,            # 文本块向量数据库
            self.chunk_entity_relation_graph,  # 实体关系图存储
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        """
        查询操作完成后的清理工作
        
        过程示例：
        1. 保存LLM响应缓存
        2. 清理临时查询数据
        3. 释放查询相关资源
        """
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
