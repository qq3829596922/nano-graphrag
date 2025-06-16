"""
GraphRAG学习系列 - 第二步：文本处理系统
==================================

这一步我们学习GraphRAG中的文本分块（Text Chunking）功能。
分块是将长文档切分成适合LLM处理的小段落，这是构建知识图谱的第二个关键步骤。

学习重点：
1. 基于Token数量的分块策略
2. 基于分隔符的智能分块
3. 重叠机制防止信息丢失
4. 批量处理优化性能
5. 与存储系统的集成

前置依赖：step1_document_processing.py
"""

import os
import json
import hashlib
import asyncio
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime


# ====================== 分块结果数据结构 ======================

@dataclass
class ChunkResult:
    """文档分块的结果数据结构"""
    content: str          # 分块的文本内容
    token_count: int      # Token数量
    chunk_index: int      # 在原文档中的分块顺序
    doc_id: str          # 原始文档ID
    chunk_id: str        # 分块的唯一ID（基于内容哈希）
    
    def to_dict(self) -> Dict:
        """转换为字典格式（与GraphRAG兼容）"""
        return {
            "tokens": self.token_count,
            "content": self.content,
            "chunk_order_index": self.chunk_index,
            "full_doc_id": self.doc_id,
        }


# ====================== 分隔符分割器 ======================

class SeparatorSplitter:
    """
    基于分隔符的智能文本分割器
    
    这个类实现了GraphRAG中的SeparatorSplitter逻辑，
    可以根据语义边界（段落、句子、词语）进行分割，
    并自动合并小片段到合适的大小。
    """
    
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,  # 分隔符的token表示
        keep_separator: Union[bool, str] = "end",      # 保留分隔符位置
        chunk_size: int = 1200,                        # 目标分块大小
        chunk_overlap: int = 100,                      # 重叠token数量
        length_function: Callable = len,               # 长度计算函数
    ):
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
    
    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        """
        核心分割方法：将token列表分割成合适大小的分块
        
        流程：
        1. 按照分隔符找到切分点
        2. 合并小片段到合理大小
        3. 添加重叠以保持上下文连贯性
        """
        # 第一步：按分隔符分割
        splits = self._split_tokens_with_separators(tokens)
        
        # 第二步：合并并优化分块大小
        return self._merge_splits(splits)
    
    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        """使用分隔符找到所有切分点"""
        splits = []
        current_split = []
        i = 0
        
        while i < len(tokens):
            separator_found = False
            
            # 检查是否匹配任何分隔符
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    # 处理分隔符的保留策略
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    
                    i += len(separator)
                    separator_found = True
                    break
            
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        
        # 添加最后一个分割
        if current_split:
            splits.append(current_split)
        
        return [s for s in splits if s]  # 过滤空分割
    
    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        """智能合并小分割到合适的分块大小"""
        if not splits:
            return []
        
        merged_splits = []
        current_chunk = []
        
        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif (self._length_function(current_chunk) + 
                  self._length_function(split) <= self._chunk_size):
                # 可以合并
                current_chunk.extend(split)
            else:
                # 当前分块已满，开始新分块
                merged_splits.append(current_chunk)
                current_chunk = split
        
        # 添加最后一个分块
        if current_chunk:
            merged_splits.append(current_chunk)
        
        # 处理过大的分块
        if (len(merged_splits) == 1 and 
            self._length_function(merged_splits[0]) > self._chunk_size):
            return self._split_large_chunk(merged_splits[0])
        
        # 添加重叠机制
        if self._chunk_overlap > 0:
            return self._add_overlap(merged_splits)
        
        return merged_splits
    
    def _split_large_chunk(self, chunk: List[int]) -> List[List[int]]:
        """将过大的分块进一步切分"""
        result = []
        step_size = self._chunk_size - self._chunk_overlap
        
        for i in range(0, len(chunk), step_size):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:  # 避免太小的分块
                result.append(new_chunk)
        
        return result
    
    def _add_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        """为分块添加重叠区域以保持上下文连贯性"""
        if len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]  # 第一个分块不需要重叠
        
        for i in range(1, len(chunks)):
            # 从前一个分块取重叠部分
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            overlap_tokens = prev_chunk[-self._chunk_overlap:]
            new_chunk = overlap_tokens + current_chunk
            
            # 确保不超过最大大小
            if self._length_function(new_chunk) > self._chunk_size:
                new_chunk = new_chunk[:self._chunk_size]
            
            result.append(new_chunk)
        
        return result


# ====================== 文本处理器主类 ======================

class TextProcessor:
    """
    GraphRAG的文本处理核心类
    
    负责将文档转换为适合后续处理的文本分块，
    支持多种分块策略和批量处理优化。
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        default_chunk_size: int = 1200,
        default_overlap_size: int = 100,
        enable_tiktoken_cache: bool = True,
    ):
        self.model_name = model_name
        self.default_chunk_size = default_chunk_size
        self.default_overlap_size = default_overlap_size
        
        # 初始化tiktoken编码器（用于精确计算token数量）
        self.encoder = tiktoken.encoding_for_model(model_name)
        
        # 默认的智能分隔符（按优先级排列）
        self.default_separators = [
            "\n\n",      # 段落分隔（最高优先级）
            "\n",        # 行分隔
            "。",        # 中文句号
            "．",        # 全角句号
            ".",         # 英文句号
            "！",        # 中文感叹号
            "!",         # 英文感叹号
            "？",        # 中文问号
            "?",         # 英文问号
            "；",        # 中文分号
            ";",         # 英文分号
            "，",        # 中文逗号
            ",",         # 英文逗号
            " ",         # 空格（最低优先级）
        ]
        
        print(f"✅ 文本处理器初始化完成")
        print(f"   - 模型: {model_name}")
        print(f"   - 默认分块大小: {default_chunk_size} tokens")
        print(f"   - 默认重叠大小: {default_overlap_size} tokens")
    
    def compute_chunk_id(self, content: str, prefix: str = "chunk-") -> str:
        """为分块内容生成唯一ID"""
        return prefix + hashlib.md5(content.encode()).hexdigest()
    
    def chunking_by_token_size(
        self,
        documents: Dict[str, str],  # {doc_id: content}
        max_token_size: Optional[int] = None,
        overlap_token_size: Optional[int] = None,
    ) -> List[ChunkResult]:
        """
        基于Token数量的文档分块方法
        
        这是GraphRAG中的经典分块策略，使用滑动窗口机制：
        - 精确控制每个分块的token数量
        - 设置重叠区域防止信息丢失
        - 使用批量编码优化性能
        
        参数：
            documents: 要处理的文档字典
            max_token_size: 最大分块token数
            overlap_token_size: 重叠token数
        
        返回：
            分块结果列表
        """
        if max_token_size is None:
            max_token_size = self.default_chunk_size
        if overlap_token_size is None:
            overlap_token_size = self.default_overlap_size
        
        print(f"🔧 开始基于Token数量的分块...")
        print(f"   - 分块大小: {max_token_size} tokens")
        print(f"   - 重叠大小: {overlap_token_size} tokens")
        print(f"   - 文档数量: {len(documents)}")
        
        results = []
        doc_ids = list(documents.keys())
        contents = list(documents.values())
        
        # 批量编码所有文档（性能优化）
        print("   - 正在批量编码文档...")
        tokens_list = self.encoder.encode_batch(contents, num_threads=16)
        
        # 处理每个文档
        for doc_index, tokens in enumerate(tokens_list):
            doc_id = doc_ids[doc_index]
            print(f"   - 处理文档 {doc_id}: {len(tokens)} tokens")
            
            chunk_tokens_list = []
            token_lengths = []
            
            # 滑动窗口分块
            step_size = max_token_size - overlap_token_size
            for start in range(0, len(tokens), step_size):
                end = start + max_token_size
                chunk_tokens = tokens[start:end]
                chunk_tokens_list.append(chunk_tokens)
                token_lengths.append(len(chunk_tokens))
            
            # 批量解码token为文本
            chunk_contents = self.encoder.decode_batch(chunk_tokens_list)
            
            # 构建分块结果
            for i, content in enumerate(chunk_contents):
                content = content.strip()
                if content:  # 只保留非空内容
                    chunk_result = ChunkResult(
                        content=content,
                        token_count=token_lengths[i],
                        chunk_index=i,
                        doc_id=doc_id,
                        chunk_id=self.compute_chunk_id(content)
                    )
                    results.append(chunk_result)
            
            print(f"     → 生成 {len(chunk_contents)} 个分块")
        
        print(f"✅ Token分块完成，共生成 {len(results)} 个分块")
        return results
    
    def chunking_by_separators(
        self,
        documents: Dict[str, str],
        max_token_size: Optional[int] = None,
        overlap_token_size: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ) -> List[ChunkResult]:
        """
        基于分隔符的智能文档分块方法
        
        这种方法更关注语义完整性：
        - 按照语义边界（段落、句子）进行分割
        - 智能合并小片段到合适大小
        - 保持上下文的连贯性
        
        参数：
            documents: 要处理的文档字典
            max_token_size: 最大分块token数
            overlap_token_size: 重叠token数
            separators: 自定义分隔符列表
        
        返回：
            分块结果列表
        """
        if max_token_size is None:
            max_token_size = self.default_chunk_size
        if overlap_token_size is None:
            overlap_token_size = self.default_overlap_size
        if separators is None:
            separators = self.default_separators
        
        print(f"🔧 开始基于分隔符的智能分块...")
        print(f"   - 分块大小: {max_token_size} tokens")
        print(f"   - 重叠大小: {overlap_token_size} tokens")
        print(f"   - 分隔符数量: {len(separators)}")
        print(f"   - 文档数量: {len(documents)}")
        
        # 将分隔符编码为token
        separator_tokens = [self.encoder.encode(s) for s in separators]
        
        # 创建分隔符分割器
        splitter = SeparatorSplitter(
            separators=separator_tokens,
            chunk_size=max_token_size,
            chunk_overlap=overlap_token_size,
        )
        
        results = []
        doc_ids = list(documents.keys())
        contents = list(documents.values())
        
        # 批量编码
        print("   - 正在批量编码文档...")
        tokens_list = self.encoder.encode_batch(contents, num_threads=16)
        
        # 处理每个文档
        for doc_index, tokens in enumerate(tokens_list):
            doc_id = doc_ids[doc_index]
            print(f"   - 处理文档 {doc_id}: {len(tokens)} tokens")
            
            # 使用分割器分块
            chunk_tokens_list = splitter.split_tokens(tokens)
            token_lengths = [len(chunk) for chunk in chunk_tokens_list]
            
            # 解码为文本
            chunk_contents = self.encoder.decode_batch(chunk_tokens_list)
            
            # 构建结果
            for i, content in enumerate(chunk_contents):
                content = content.strip()
                if content:
                    chunk_result = ChunkResult(
                        content=content,
                        token_count=token_lengths[i],
                        chunk_index=i,
                        doc_id=doc_id,
                        chunk_id=self.compute_chunk_id(content)
                    )
                    results.append(chunk_result)
            
            print(f"     → 生成 {len(chunk_contents)} 个分块")
        
        print(f"✅ 分隔符分块完成，共生成 {len(results)} 个分块")
        return results
    
    def export_chunks_to_graphrag_format(
        self, 
        chunks: List[ChunkResult]
    ) -> Dict[str, Dict]:
        """
        将分块结果导出为GraphRAG兼容的格式
        
        格式：{chunk_id: chunk_data}
        """
        result = {}
        for chunk in chunks:
            result[chunk.chunk_id] = chunk.to_dict()
        return result
    
    async def save_chunks_to_storage(
        self,
        chunks: List[ChunkResult],
        storage_dir: str = "step2_chunks_storage"
    ):
        """将分块结果保存到存储系统（简化版存储）"""
        # 创建存储目录
        storage_path = Path(storage_dir)
        storage_path.mkdir(exist_ok=True)
        
        # 转换为GraphRAG格式
        chunk_data = self.export_chunks_to_graphrag_format(chunks)
        
        # 保存为JSON文件
        chunks_file = storage_path / "text_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存 {len(chunks)} 个分块到存储系统")
        print(f"   - 存储位置: {chunks_file}")
        
        return chunk_data


# ====================== 演示和测试 ======================

async def demo_text_processing():
    """演示文本处理功能"""
    print("=" * 60)
    print("GraphRAG学习系列 - 第二步：文本处理系统演示")
    print("=" * 60)
    
    # 准备测试文档（更丰富的内容）
    test_documents = {
        "doc-ai-overview": """
        人工智能概述
        
        人工智能（Artificial Intelligence，AI）是计算机科学的一个重要分支，致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。
        
        AI的发展历程可以追溯到20世纪50年代。1956年，约翰·麦卡锡首次提出了"人工智能"这一术语。从那时起，AI经历了多次起伏，包括两次AI寒冬期。
        
        机器学习的兴起
        
        机器学习（Machine Learning，ML）是AI的一个核心分支，它使计算机能够在没有明确编程的情况下进行学习。机器学习算法通过数据进行训练，能够识别模式并做出预测。
        
        深度学习革命
        
        深度学习（Deep Learning，DL）是机器学习的一个子领域，基于人工神经网络进行学习。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。
        
        现代AI应用
        
        今天的AI技术已经广泛应用于各个领域：搜索引擎、推荐系统、自动驾驶、医疗诊断、金融风控等。ChatGPT等大语言模型的出现，更是将AI技术推向了新的高度。
        """,
        
        "doc-graphrag-intro": """
        GraphRAG技术详解
        
        GraphRAG（Graph Retrieval-Augmented Generation）是一种创新的RAG架构，它结合了知识图谱和检索增强生成技术。
        
        传统RAG的局限性
        
        传统的RAG系统主要依赖向量相似度进行检索，这种方法在处理复杂问题时存在局限：
        1. 难以捕获实体间的复杂关系
        2. 缺乏全局性的推理能力
        3. 对于需要多跳推理的问题效果不佳
        
        GraphRAG的创新
        
        GraphRAG通过构建知识图谱来解决这些问题：
        - 实体提取：从文档中识别命名实体
        - 关系提取：挖掘实体间的语义关系
        - 图谱构建：将实体和关系组织成图结构
        - 社区检测：发现实体集群和主题
        
        查询处理机制
        
        GraphRAG支持三种查询模式：
        1. Local查询：基于局部子图的详细分析
        2. Global查询：基于社区摘要的全局理解
        3. Naive查询：传统的向量检索方式
        
        技术优势
        
        相比传统RAG，GraphRAG具有以下优势：更好的推理能力、更强的可解释性、更准确的答案生成。
        """
    }
    
    # 创建文本处理器
    processor = TextProcessor(
        default_chunk_size=400,   # 使用较小的分块便于演示
        default_overlap_size=50
    )
    
    print("\n" + "="*50)
    print("📝 测试1: 基于Token数量的分块")
    print("="*50)
    
    token_chunks = processor.chunking_by_token_size(test_documents)
    
    print(f"\n📊 分块统计:")
    for i, chunk in enumerate(token_chunks):
        print(f"分块 {i+1}:")
        print(f"  - 来源文档: {chunk.doc_id}")
        print(f"  - Token数量: {chunk.token_count}")
        print(f"  - 分块顺序: {chunk.chunk_index}")
        print(f"  - 分块ID: {chunk.chunk_id}")
        print(f"  - 内容预览: {chunk.content[:80]}...")
        print()
    
    print("\n" + "="*50)
    print("📝 测试2: 基于分隔符的智能分块")
    print("="*50)
    
    separator_chunks = processor.chunking_by_separators(test_documents)
    
    print(f"\n📊 分块统计:")
    for i, chunk in enumerate(separator_chunks):
        print(f"分块 {i+1}:")
        print(f"  - 来源文档: {chunk.doc_id}")
        print(f"  - Token数量: {chunk.token_count}")
        print(f"  - 分块顺序: {chunk.chunk_index}")
        print(f"  - 分块ID: {chunk.chunk_id}")
        print(f"  - 内容预览: {chunk.content[:80]}...")
        print()
    
    print("\n" + "="*50)
    print("📝 测试3: 与存储系统集成")
    print("="*50)
    
    # 保存分块到存储系统
    chunk_data = await processor.save_chunks_to_storage(token_chunks)
    
    print(f"\n📊 存储格式预览:")
    for chunk_id, chunk_info in list(chunk_data.items())[:2]:  # 只显示前2个
        print(f"分块ID: {chunk_id}")
        print(f"数据结构: {json.dumps(chunk_info, ensure_ascii=False, indent=2)}")
        print()
    
    print("\n" + "="*50)
    print("📝 测试4: 性能对比分析")
    print("="*50)
    
    import time
    
    # Token分块性能测试
    start_time = time.time()
    token_chunks = processor.chunking_by_token_size(test_documents)
    token_time = time.time() - start_time
    
    # 分隔符分块性能测试
    start_time = time.time()
    separator_chunks = processor.chunking_by_separators(test_documents)
    separator_time = time.time() - start_time
    
    print(f"⚡ 性能对比:")
    print(f"  - Token分块: {token_time:.4f}s, 生成 {len(token_chunks)} 个分块")
    print(f"  - 分隔符分块: {separator_time:.4f}s, 生成 {len(separator_chunks)} 个分块")
    print(f"  - 性能比值: {separator_time/token_time:.2f}x")
    
    print("\n" + "="*50)
    print("✅ 文本处理系统演示完成！")
    print("="*50)
    print("\n下一步学习:")
    print("- step3_entity_extraction.py: 实体提取系统")
    print("- 从文本分块中识别实体和关系")
    print("- 使用LLM进行智能解析")


def analyze_chunking_strategies():
    """分析不同分块策略的特点"""
    print("\n" + "="*60)
    print("📋 分块策略对比分析")
    print("="*60)
    
    comparison_data = [
        ["特性", "Token分块", "分隔符分块"],
        ["精确性", "精确控制大小", "语义边界优先"],
        ["语义完整性", "可能截断句子", "保持语义完整"],
        ["处理速度", "快", "稍慢（需要分隔符匹配）"],
        ["内存使用", "低", "中等"],
        ["适用场景", "严格大小要求", "语义质量要求"],
        ["重叠处理", "简单滑动窗口", "智能边界对齐"],
        ["可定制性", "参数简单", "可定制分隔符规则"],
    ]
    
    # 打印表格
    for row in comparison_data:
        print(f"| {row[0]:<12} | {row[1]:<15} | {row[2]:<20} |")
        if row[0] == "特性":
            print("|" + "-"*14 + "|" + "-"*17 + "|" + "-"*22 + "|")
    
    print(f"\n💡 选择建议:")
    print(f"  - 对于严格控制分块大小的场景，选择Token分块")
    print(f"  - 对于注重语义完整性的场景，选择分隔符分块")
    print(f"  - 对于混合需求，可以组合使用两种策略")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_text_processing())
    
    # 分析分块策略
    analyze_chunking_strategies()