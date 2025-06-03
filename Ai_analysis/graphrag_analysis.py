"""
GraphRAG 初始化过程详细分析
这个文件详细分析了 graph_func = GraphRAG(working_dir="./mytest") 的执行过程
"""
import os
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken
import asyncio

# 导入模拟的相关模块（实际代码中这些是从其他文件导入的）
from nano_graphrag._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
)
from nano_graphrag._op import chunking_by_token_size
from nano_graphrag._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from nano_graphrag._utils import (
    EmbeddingFunc,
    limit_async_func_call,
    convert_response_to_json,
    logger,
)
from nano_graphrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    QueryParam,
)

@dataclass
class GraphRAG:
    """
    GraphRAG 主类 - 负责管理整个图检索增强生成系统
    这是一个数据类，使用 @dataclass 装饰器自动生成构造函数
    """
    
    # ===== 核心配置参数 =====
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )  # 工作目录，默认使用当前时间戳创建唯一目录
    
    # ===== 功能开关 =====
    enable_local: bool = True      # 是否启用本地查询模式
    enable_naive_rag: bool = False # 是否启用朴素RAG模式
    
    # ===== 文本分块配置 =====
    chunk_func: Callable = chunking_by_token_size  # 文本分块函数
    chunk_token_size: int = 1200                   # 每个文本块的token大小
    chunk_overlap_token_size: int = 100            # 文本块重叠的token数量
    tiktoken_model_name: str = "gpt-4o"            # 用于计算token的模型名称
    
    # ===== 实体提取配置 =====
    entity_extract_max_gleaning: int = 1     # 实体提取的最大gleaning次数
    entity_summary_to_max_tokens: int = 500  # 实体摘要的最大token数
    
    # ===== 图聚类配置 =====
    graph_cluster_algorithm: str = "leiden"   # 图聚类算法（Leiden算法）
    max_graph_cluster_size: int = 10          # 最大图聚类大小
    graph_cluster_seed: int = 0xDEADBEEF      # 图聚类的随机种子
    
    # ===== 节点嵌入配置 =====
    node_embedding_algorithm: str = "node2vec"  # 节点嵌入算法
    node2vec_params: dict = field(              # node2vec算法的参数
        default_factory=lambda: {
            "dimensions": 1536,     # 嵌入维度
            "num_walks": 10,        # 随机游走次数
            "walk_length": 40,      # 游走长度
            "window_size": 2,       # 窗口大小
            "iterations": 3,        # 迭代次数
            "random_seed": 3,       # 随机种子
        }
    )
    
    # ===== 文本嵌入配置 =====
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)  # 嵌入函数
    embedding_batch_num: int = 32           # 嵌入批处理数量
    embedding_func_max_async: int = 16      # 嵌入函数最大异步并发数
    query_better_than_threshold: float = 0.2  # 查询相似度阈值
    
    # ===== LLM配置 =====
    using_azure_openai: bool = False        # 是否使用Azure OpenAI
    using_amazon_bedrock: bool = False      # 是否使用Amazon Bedrock
    best_model_func: callable = gpt_4o_complete      # 最佳模型函数
    best_model_max_token_size: int = 32768            # 最佳模型最大token数
    best_model_max_async: int = 16                    # 最佳模型最大异步并发数
    cheap_model_func: callable = gpt_4o_mini_complete # 便宜模型函数
    cheap_model_max_token_size: int = 32768           # 便宜模型最大token数
    cheap_model_max_async: int = 16                   # 便宜模型最大异步并发数
    
    # ===== 存储配置 =====
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage      # JSON键值存储类
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage       # 向量数据库存储类
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)           # 向量数据库存储参数
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage                # 图存储类
    
    # ===== 其他配置 =====
    enable_llm_cache: bool = True                     # 是否启用LLM缓存
    always_create_working_dir: bool = True            # 是否总是创建工作目录
    addon_params: dict = field(default_factory=dict)  # 附加参数
    convert_response_to_json_func: callable = convert_response_to_json  # 响应转JSON函数

    def __post_init__(self):
        """
        数据类的后初始化方法，在 __init__ 之后自动调用
        这里进行所有的初始化逻辑
        """
        print("=== GraphRAG 初始化开始 ===")
        
        # 1. 打印配置信息
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")
        print(f"步骤1: 加载配置参数完成，工作目录设置为: {self.working_dir}")
        
        # 2. 根据配置选择合适的LLM提供商
        if self.using_azure_openai:
            print("步骤2: 检测到使用 Azure OpenAI，切换相关函数...")
            # 这里会切换到Azure版本的函数（代码中有具体实现）
            logger.info("Switched to Azure OpenAI functions")
            
        if self.using_amazon_bedrock:
            print("步骤2: 检测到使用 Amazon Bedrock，切换相关函数...")
            # 这里会切换到Bedrock版本的函数（代码中有具体实现）
            logger.info("Switched to Amazon Bedrock functions")
        
        # 3. 创建工作目录
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            print(f"步骤3: 创建工作目录 {self.working_dir}")
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        else:
            print(f"步骤3: 工作目录 {self.working_dir} 已存在或不需要创建")
        
        # 4. 初始化各种存储组件
        print("步骤4: 初始化存储组件...")
        
        # 4.1 完整文档存储 - 存储原始输入文档
        print("  4.1: 初始化完整文档存储 (full_docs)")
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",           # 命名空间，用于区分不同类型的数据
            global_config=asdict(self)       # 全局配置，传递给存储类
        )
        # 实际创建文件: ./mytest/kv_store_full_docs.json
        
        # 4.2 文本块存储 - 存储分块后的文本
        print("  4.2: 初始化文本块存储 (text_chunks)")
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self)
        )
        # 实际创建文件: ./mytest/kv_store_text_chunks.json
        
        # 4.3 LLM响应缓存存储 - 缓存LLM的响应以避免重复调用
        print("  4.3: 初始化LLM响应缓存存储 (llm_response_cache)")
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self)
            )
            if self.enable_llm_cache  # 只有启用缓存时才创建
            else None
        )
        # 实际创建文件: ./mytest/kv_store_llm_response_cache.json（如果启用缓存）
        
        # 4.4 社区报告存储 - 存储图聚类后生成的社区报告
        print("  4.4: 初始化社区报告存储 (community_reports)")
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports",
            global_config=asdict(self)
        )
        # 实际创建文件: ./mytest/kv_store_community_reports.json
        
        # 4.5 图存储 - 存储实体关系图
        print("  4.5: 初始化图存储 (chunk_entity_relation_graph)")
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self)
        )
        # 实际创建文件: ./mytest/graph_chunk_entity_relation.graphml
        
        # 5. 设置嵌入函数的并发限制
        print("步骤5: 设置嵌入函数并发限制...")
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        
        # 6. 初始化向量数据库（如果启用本地查询）
        if self.enable_local:
            print("  6.1: 初始化实体向量数据库 (entities_vdb)")
            self.entities_vdb = self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,  # 传入嵌入函数
                meta_fields={"entity_name"},         # 元数据字段
            )
            # 实际创建文件: ./mytest/vdb_entities.json
        else:
            self.entities_vdb = None
            print("  6.1: 本地查询未启用，跳过实体向量数据库初始化")
        
        # 6.2 初始化朴素RAG向量数据库（如果启用）
        if self.enable_naive_rag:
            print("  6.2: 初始化朴素RAG向量数据库 (chunks_vdb)")
            self.chunks_vdb = self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            # 实际创建文件: ./mytest/vdb_chunks.json
        else:
            self.chunks_vdb = None
            print("  6.2: 朴素RAG未启用，跳过文本块向量数据库初始化")
        
        # 7. 设置LLM函数的并发限制和缓存
        print("步骤7: 设置LLM函数并发限制和缓存...")
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )
        
        print("=== GraphRAG 初始化完成 ===")
        print(f"工作目录: {self.working_dir}")
        print("已创建的存储组件:")
        print("  - 完整文档存储 (JSON)")
        print("  - 文本块存储 (JSON)")
        print("  - LLM响应缓存 (JSON)")
        print("  - 社区报告存储 (JSON)")
        print("  - 图存储 (GraphML)")
        if self.enable_local:
            print("  - 实体向量数据库 (NanoVectorDB)")
        if self.enable_naive_rag:
            print("  - 文本块向量数据库 (NanoVectorDB)")

    def insert(self, string_or_strings):
        """插入文档的同步方法"""
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """查询的同步方法"""
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    def _get_or_create_event_loop(self):
        """获取或创建事件循环"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.info("Creating a new event loop in a sub-thread.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    async def ainsert(self, string_or_strings):
        """异步插入文档方法（简化版，实际实现更复杂）"""
        print("开始异步插入文档...")
        # 实际的插入逻辑包括：
        # 1. 文档去重
        # 2. 文本分块
        # 3. 实体提取
        # 4. 图构建
        # 5. 社区检测
        # 6. 向量化存储
        pass

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """异步查询方法（简化版，实际实现更复杂）"""
        print(f"开始异步查询: {query}")
        print(f"查询模式: {param.mode}")
        # 实际的查询逻辑根据模式不同：
        # - local: 使用实体向量数据库进行本地查询
        # - global: 使用社区报告进行全局查询
        # - naive: 使用简单的向量相似度查询
        pass


if __name__ == "__main__":
    # 演示 GraphRAG 初始化过程
    print("开始演示 GraphRAG 初始化过程...\n")
    
    # 这行代码等价于你的 graph_func = GraphRAG(working_dir="./mytest")
    graph_func = GraphRAG(working_dir="./mytest")
    
    print(f"\n初始化完成！GraphRAG 实例已创建")
    print(f"类型: {type(graph_func)}")
    print(f"工作目录: {graph_func.working_dir}") 