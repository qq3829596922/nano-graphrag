"""
GraphRAG insert 方法执行流程详细分析
这个文件详细分析了 graph_func.insert(f.read()) 的完整执行过程

基于运行日志和源码分析，insert 方法执行了以下6个主要步骤：
1. 文档预处理和去重
2. 文本分块处理
3. 实体和关系提取
4. 知识图谱构建
5. 社区检测和报告生成
6. 数据持久化存储
"""

import asyncio
import os
from dataclasses import asdict
from functools import partial
from typing import Dict, List, Union, cast

# 模拟导入（实际代码中这些是从其他模块导入的）
from nano_graphrag._utils import (
    compute_mdhash_id, 
    always_get_an_event_loop,
    logger
)
from nano_graphrag._op import (
    get_chunks,
    extract_entities,
    generate_community_report
)
from nano_graphrag.base import StorageNameSpace


class GraphRAGInsertAnalysis:
    """
    GraphRAG insert 方法的详细分析类
    """
    
    def insert(self, string_or_strings):
        """
        同步的 insert 方法入口点
        
        这是用户调用的主要方法：graph_func.insert(f.read())
        
        执行流程：
        1. 获取或创建事件循环
        2. 运行异步的 ainsert 方法
        
        Args:
            string_or_strings: 可以是单个字符串或字符串列表
        """
        print("=== INSERT 方法开始执行 ===")
        print(f"输入数据类型: {type(string_or_strings)}")
        
        # 获取异步事件循环，确保可以运行异步代码
        loop = always_get_an_event_loop()
        
        # 运行异步的实际插入方法
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        """
        异步的核心 insert 实现方法
        
        这是整个插入过程的核心，包含6个主要步骤
        """
        print("\n=== 异步插入流程开始 ===")
        
        # 开始插入前的准备工作
        await self._insert_start()
        
        try:
            # ========== 步骤1: 文档预处理和去重 ==========
            print("\n--- 步骤1: 文档预处理和去重 ---")
            
            # 1.1 标准化输入格式（确保是列表）
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
                print("输入转换为列表格式")
            
            # 1.2 为每个文档生成唯一的哈希ID
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            print(f"生成 {len(new_docs)} 个文档，每个都有唯一的哈希ID")
            
            # 1.3 检查文档是否已存在（去重）
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            
            if not len(new_docs):
                logger.warning("所有文档都已存在于存储中，跳过处理")
                return
            
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")
            print(f"确认插入 {len(new_docs)} 个新文档")

            # ========== 步骤2: 文本分块处理 ==========
            print("\n--- 步骤2: 文本分块处理 ---")
            
            # 2.1 调用分块函数，将长文档切分为小块
            inserting_chunks = get_chunks(
                new_docs=new_docs,                              # 输入文档
                chunk_func=self.chunk_func,                     # 分块函数（默认: chunking_by_token_size）
                overlap_token_size=self.chunk_overlap_token_size, # 重叠token数（默认: 100）
                max_token_size=self.chunk_token_size,           # 最大分块大小（默认: 1200）
            )
            print(f"文档分块完成，生成 {len(inserting_chunks)} 个文本块")
            
            # 2.2 检查分块是否已存在（去重）
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            
            if not len(inserting_chunks):
                logger.warning("所有文本块都已存在于存储中，跳过处理")
                return
            
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            print(f"确认插入 {len(inserting_chunks)} 个新文本块")
            
            # 2.3 如果启用朴素RAG，将分块存储到向量数据库
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)
                print("文本块已存储到朴素RAG向量数据库")

            # 2.4 清空现有的社区报告（因为图结构可能发生变化）
            await self.community_reports.drop()
            print("清空现有社区报告，准备重新生成")

            # ========== 步骤3: 实体和关系提取 ==========
            print("\n--- 步骤3: 实体和关系提取 ---")
            
            logger.info("[Entity Extraction]...")
            
            # 3.1 使用LLM从文本块中提取实体和关系
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,                               # 输入的文本块
                knwoledge_graph_inst=self.chunk_entity_relation_graph, # 图存储实例
                entity_vdb=self.entities_vdb,                  # 实体向量数据库
                global_config=asdict(self),                    # 全局配置
                using_amazon_bedrock=self.using_amazon_bedrock # 是否使用Amazon Bedrock
            )
            
            if maybe_new_kg is None:
                logger.warning("未找到新实体，跳过后续处理")
                return
            
            # 3.2 更新知识图谱实例
            self.chunk_entity_relation_graph = maybe_new_kg
            print("实体和关系提取完成，知识图谱已更新")

            # ========== 步骤4: 图聚类分析 ==========
            print("\n--- 步骤4: 图聚类分析 ---")
            
            logger.info("[Community Report]...")
            
            # 4.1 对知识图谱进行聚类，发现社区结构
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm  # 默认使用 Leiden 算法
            )
            print(f"图聚类完成，使用算法: {self.graph_cluster_algorithm}")

            # ========== 步骤5: 社区报告生成 ==========
            print("\n--- 步骤5: 社区报告生成 ---")
            
            # 5.1 为每个社区生成描述性报告
            await generate_community_report(
                self.community_reports,                # 社区报告存储
                self.chunk_entity_relation_graph,     # 知识图谱
                asdict(self)                          # 全局配置
            )
            print("社区报告生成完成")

            # ========== 步骤6: 数据持久化存储 ==========
            print("\n--- 步骤6: 数据持久化存储 ---")
            
            # 6.1 保存原始文档
            await self.full_docs.upsert(new_docs)
            print("原始文档已保存")
            
            # 6.2 保存文本块
            await self.text_chunks.upsert(inserting_chunks)
            print("文本块已保存")
            
            print("\n=== INSERT 处理完成 ===")
            
        finally:
            # 无论成功或失败，都要执行清理工作
            await self._insert_done()

    async def _insert_start(self):
        """
        插入开始前的准备工作
        
        主要是初始化各个存储组件的索引
        """
        print("准备工作: 初始化存储组件索引...")
        
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,  # 图存储
        ]:
            if storage_inst is None:
                continue
            # 调用每个存储组件的索引开始回调
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        
        # 并行执行所有准备任务
        await asyncio.gather(*tasks)
        print("存储组件索引初始化完成")

    async def _insert_done(self):
        """
        插入完成后的清理工作
        
        主要是保存所有存储组件的数据到磁盘
        """
        print("清理工作: 保存所有数据到磁盘...")
        
        tasks = []
        for storage_inst in [
            self.full_docs,                     # 完整文档存储
            self.text_chunks,                   # 文本块存储
            self.llm_response_cache,           # LLM响应缓存
            self.community_reports,            # 社区报告存储
            self.entities_vdb,                 # 实体向量数据库
            self.chunks_vdb,                   # 文本块向量数据库
            self.chunk_entity_relation_graph,  # 图存储
        ]:
            if storage_inst is None:
                continue
            # 调用每个存储组件的索引完成回调（保存数据）
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        
        # 并行执行所有保存任务
        await asyncio.gather(*tasks)
        print("所有数据已保存到磁盘")


def analyze_insert_execution():
    """
    分析实际运行日志中的 insert 执行过程
    """
    print("=== 基于运行日志的 INSERT 执行分析 ===\n")
    
    execution_log = """
    根据实际运行日志，insert 方法的执行过程如下：
    
    1. 初始化阶段:
       - Load KV full_docs with 0 data (加载文档存储，当前为空)
       - Load KV text_chunks with 0 data (加载文本块存储，当前为空)
       - Load KV llm_response_cache with 0 data (加载LLM缓存，当前为空)
       - Load KV community_reports with 0 data (加载社区报告存储，当前为空)
       - Init vector database for entities (初始化实体向量数据库)
    
    2. 文档处理阶段:
       - [New Docs] inserting 1 docs (插入1个新文档)
       - [New Chunks] inserting 3 chunks (生成3个文本块)
    
    3. 实体提取阶段:
       - [Entity Extraction]... (开始实体提取)
       - 发送多个 HTTP 请求到 LLM API (https://xiaoai.plus/v1/chat/completions)
       - Processed 3(100%) chunks, 53 entities(duplicated), 28 relations(duplicated)
       - 从3个文本块中提取了53个实体和28个关系
    
    4. 向量化阶段:
       - Inserting 43 vectors to entities (向实体向量数据库插入43个向量)
       - 发送嵌入请求到 API (https://xiaoai.plus/v1/embeddings)
    
    5. 社区分析阶段:
       - [Community Report]... (开始社区报告生成)
       - Each level has communities: {0: 4} (检测到4个社区)
       - Generating by levels: [0] (为社区生成报告)
       - 发送多个请求生成社区描述
       - Processed 4 communities (处理了4个社区)
    
    6. 图存储阶段:
       - Writing graph with 43 nodes, 27 edges (保存包含43个节点、27条边的图)
    
    总结：
    - 输入：1个文档（三国演义片段）
    - 分块：3个文本块
    - 实体：43个（去重后）
    - 关系：27条（去重后）
    - 社区：4个
    - API调用：约12次LLM调用 + 2次嵌入调用
    """
    
    print(execution_log)


def analyze_api_calls():
    """
    分析 insert 过程中的 API 调用
    """
    print("\n=== API 调用分析 ===\n")
    
    api_analysis = """
    根据日志，insert 过程中进行了以下 API 调用：
    
    1. 实体提取阶段 (6次 chat/completions 调用):
       - 为每个文本块调用 LLM 提取实体和关系
       - 可能包括实体总结和关系验证
    
    2. 向量化阶段 (2次 embeddings 调用):
       - 将提取的实体转换为向量表示
       - 批量处理以提高效率
    
    3. 社区报告生成阶段 (4次 chat/completions 调用):
       - 为每个检测到的社区生成描述性报告
       - 使用 JSON 格式的结构化输出
    
    4. 查询执行阶段 (2次 chat/completions 调用):
       - 执行实际的查询请求
       - 生成最终的回答
    
    总计 API 调用：
    - LLM 文本生成：12次
    - 向量嵌入：2次
    - 总计：14次 API 调用
    
    成本估算（假设使用 GPT-4）：
    - 输入 token：约 10,000-15,000
    - 输出 token：约 5,000-8,000
    - 嵌入 token：约 2,000-3,000
    """
    
    print(api_analysis)


def analyze_storage_files():
    """
    分析 insert 过程中创建的存储文件
    """
    print("\n=== 存储文件分析 ===\n")
    
    storage_analysis = """
    insert 过程完成后，在 ./mytest/ 目录下创建/更新了以下文件：
    
    1. kv_store_full_docs.json (8.0KB)
       - 存储原始文档内容
       - 包含文档ID和完整文本
    
    2. kv_store_text_chunks.json (9.1KB)  
       - 存储分块后的文本片段
       - 包含每个分块的内容、token数量、索引等信息
    
    3. kv_store_llm_response_cache.json (28KB)
       - 缓存LLM的响应结果
       - 避免重复调用相同的请求，节省成本
    
    4. kv_store_community_reports.json (18KB)
       - 存储社区分析报告
       - 包含每个社区的描述和关键信息
    
    5. graph_chunk_entity_relation.graphml (22KB)
       - 存储实体关系图
       - 使用标准的 GraphML 格式
       - 包含43个节点（实体）和27条边（关系）
    
    6. vdb_entities.json (267KB)
       - 存储实体的向量表示
       - 包含43个实体向量，每个1536维
       - 用于语义相似度搜索
    
    总存储空间：约 370KB
    """
    
    print(storage_analysis)


if __name__ == "__main__":
    # 运行分析
    analyze_insert_execution()
    analyze_api_calls() 
    analyze_storage_files()
    
    print("\n=== 总结 ===")
    print("""
    insert 方法是 GraphRAG 的核心功能，它将原始文档转换为结构化的知识图谱。
    
    主要特点：
    1. 智能分块：根据 token 数量和重叠度分割文档
    2. 实体提取：使用 LLM 识别实体和关系
    3. 图构建：将实体关系构建为图结构
    4. 社区检测：发现实体集群和社区结构
    5. 向量化：支持语义搜索和相似度计算
    6. 缓存优化：避免重复的 API 调用
    
    这个过程为后续的查询操作奠定了基础，使系统能够进行：
    - 本地查询（基于实体向量相似度）
    - 全局查询（基于社区报告）
    - 朴素RAG查询（基于文本块相似度）
    """) 