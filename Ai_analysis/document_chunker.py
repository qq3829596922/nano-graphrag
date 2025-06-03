"""
独立的文档分块模块
从 nano-graphrag 中提取的核心文档分块功能

这个模块提供了两种文档分块方法：
1. 基于Token数量的分块 (chunking_by_token_size)
2. 基于分隔符的分块 (chunking_by_separator)

使用示例:
    chunker = DocumentChunker()
    chunks = chunker.chunk_text("你的文档内容", method="token")
"""

import re
import hashlib
import tiktoken
from typing import List, Dict, Union, Optional, Callable
from dataclasses import dataclass


@dataclass
class ChunkResult:
    """分块结果的数据结构"""
    content: str           # 分块内容
    token_count: int       # Token数量
    chunk_index: int       # 分块索引
    doc_id: str           # 文档ID
    hash_id: str          # 分块哈希ID


class SeparatorSplitter:
    """
    基于分隔符的文本分割器
    支持多个分隔符，并可以控制分块大小和重叠
    """
    
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, str] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable = len,
    ):
        """
        初始化分割器
        
        Args:
            separators: 分隔符的token列表，如[tiktoken.encode("\n"), tiktoken.encode(".")]
            keep_separator: 是否保留分隔符，"start"/"end"/True/False
            chunk_size: 分块大小（token数量）
            chunk_overlap: 分块重叠大小
            length_function: 计算长度的函数
        """
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        """将token列表按分隔符分割并合并为合适大小的分块"""
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        """使用分隔符分割token"""
        splits = []
        current_split = []
        i = 0
        
        while i < len(tokens):
            separator_found = False
            # 检查是否匹配任何分隔符
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    # 根据设置决定是否保留分隔符
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
        
        if current_split:
            splits.append(current_split)
        
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        """将小的分割合并为合适大小的分块"""
        if not splits:
            return []

        merged_splits = []
        current_chunk = []

        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                current_chunk.extend(split)
            else:
                merged_splits.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_splits.append(current_chunk)

        # 如果分块太大，需要进一步切分
        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])

        # 添加重叠
        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        
        return merged_splits

    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        """将过大的分块进一步切分"""
        result = []
        for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:
                result.append(new_chunk)
        return result

    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        """为分块添加重叠部分"""
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                # 从前一个分块取重叠部分
                overlap = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap + chunk
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[:self._chunk_size]
                result.append(new_chunk)
        return result


class DocumentChunker:
    """
    文档分块器主类
    提供多种分块方法的统一接口
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o",
        default_chunk_size: int = 1200,
        default_overlap_size: int = 100
    ):
        """
        初始化文档分块器
        
        Args:
            model_name: tiktoken模型名称，用于token计算
            default_chunk_size: 默认分块大小（token数量）  
            default_overlap_size: 默认重叠大小（token数量）
        """
        self.model_name = model_name
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.default_chunk_size = default_chunk_size
        self.default_overlap_size = default_overlap_size
        
        # 默认的文本分隔符
        self.default_separators = [
            "\n\n",      # 段落分隔
            "\n",        # 行分隔  
            "。",        # 中文句号
            ".",         # 英文句号
            "！",        # 中文感叹号
            "!",         # 英文感叹号
            "？",        # 中文问号
            "?",         # 英文问号
            "；",        # 中文分号
            ";",         # 英文分号
            " ",         # 空格
        ]

    def compute_hash_id(self, content: str, prefix: str = "chunk-") -> str:
        """为内容生成哈希ID"""
        return prefix + hashlib.md5(content.encode()).hexdigest()

    def chunking_by_token_size(
        self,
        texts: List[str],
        doc_ids: List[str],
        max_token_size: int = None,
        overlap_token_size: int = None,
    ) -> List[ChunkResult]:
        """
        基于Token数量的文档分块方法
        
        Args:
            texts: 要分块的文本列表
            doc_ids: 对应的文档ID列表
            max_token_size: 最大分块token数（默认使用初始化时的值）
            overlap_token_size: 重叠token数（默认使用初始化时的值）
            
        Returns:
            分块结果列表
        """
        if max_token_size is None:
            max_token_size = self.default_chunk_size
        if overlap_token_size is None:
            overlap_token_size = self.default_overlap_size
            
        print(f"开始基于Token数量的分块...")
        print(f"分块大小: {max_token_size} tokens, 重叠大小: {overlap_token_size} tokens")
        
        results = []
        
        # 批量编码所有文本为token
        tokens_list = self.encoder.encode_batch(texts, num_threads=16)
        
        for doc_index, tokens in enumerate(tokens_list):
            doc_id = doc_ids[doc_index]
            print(f"处理文档 {doc_id}, 总token数: {len(tokens)}")
            
            chunk_tokens = []
            lengths = []
            
            # 滑动窗口分块
            for start in range(0, len(tokens), max_token_size - overlap_token_size):
                end = start + max_token_size
                chunk_token = tokens[start:end]
                chunk_tokens.append(chunk_token)
                lengths.append(len(chunk_token))
            
            # 批量解码token为文本
            chunk_contents = self.encoder.decode_batch(chunk_tokens)
            
            # 构建结果
            for i, content in enumerate(chunk_contents):
                content = content.strip()
                if content:  # 只保留非空内容
                    chunk_result = ChunkResult(
                        content=content,
                        token_count=lengths[i],
                        chunk_index=i,
                        doc_id=doc_id,
                        hash_id=self.compute_hash_id(content)
                    )
                    results.append(chunk_result)
            
            print(f"文档 {doc_id} 分成了 {len(chunk_contents)} 个分块")
        
        print(f"分块完成，总共生成 {len(results)} 个分块")
        return results

    def chunking_by_separator(
        self,
        texts: List[str],
        doc_ids: List[str],
        max_token_size: int = None,
        overlap_token_size: int = None,
        separators: List[str] = None,
    ) -> List[ChunkResult]:
        """
        基于分隔符的文档分块方法
        
        Args:
            texts: 要分块的文本列表
            doc_ids: 对应的文档ID列表  
            max_token_size: 最大分块token数
            overlap_token_size: 重叠token数
            separators: 自定义分隔符列表
            
        Returns:
            分块结果列表
        """
        if max_token_size is None:
            max_token_size = self.default_chunk_size
        if overlap_token_size is None:
            overlap_token_size = self.default_overlap_size
        if separators is None:
            separators = self.default_separators
            
        print(f"开始基于分隔符的分块...")
        print(f"分块大小: {max_token_size} tokens, 重叠大小: {overlap_token_size} tokens")
        print(f"使用分隔符: {separators}")
        
        # 将分隔符编码为token
        separator_tokens = [self.encoder.encode(s) for s in separators]
        
        # 创建分割器
        splitter = SeparatorSplitter(
            separators=separator_tokens,
            chunk_size=max_token_size,
            chunk_overlap=overlap_token_size,
        )
        
        results = []
        tokens_list = self.encoder.encode_batch(texts, num_threads=16)
        
        for doc_index, tokens in enumerate(tokens_list):
            doc_id = doc_ids[doc_index]
            print(f"处理文档 {doc_id}, 总token数: {len(tokens)}")
            
            # 使用分割器分块
            chunk_tokens = splitter.split_tokens(tokens)
            lengths = [len(chunk) for chunk in chunk_tokens]
            
            # 解码为文本
            chunk_contents = self.encoder.decode_batch(chunk_tokens)
            
            # 构建结果
            for i, content in enumerate(chunk_contents):
                content = content.strip()
                if content:
                    chunk_result = ChunkResult(
                        content=content,
                        token_count=lengths[i],
                        chunk_index=i,
                        doc_id=doc_id,
                        hash_id=self.compute_hash_id(content)
                    )
                    results.append(chunk_result)
            
            print(f"文档 {doc_id} 分成了 {len(chunk_contents)} 个分块")
        
        print(f"分块完成，总共生成 {len(results)} 个分块")
        return results

    def chunk_text(
        self, 
        text: str, 
        method: str = "token",
        doc_id: str = None,
        **kwargs
    ) -> List[ChunkResult]:
        """
        单个文本的分块方法（便利方法）
        
        Args:
            text: 要分块的文本
            method: 分块方法，"token" 或 "separator"
            doc_id: 文档ID，如果不提供会自动生成
            **kwargs: 其他参数传递给分块方法
            
        Returns:
            分块结果列表
        """
        if doc_id is None:
            doc_id = self.compute_hash_id(text, prefix="doc-")
        
        if method == "token":
            return self.chunking_by_token_size([text], [doc_id], **kwargs)
        elif method == "separator":
            return self.chunking_by_separator([text], [doc_id], **kwargs)
        else:
            raise ValueError(f"不支持的分块方法: {method}. 请使用 'token' 或 'separator'")

    def chunk_documents(
        self,
        documents: Dict[str, str],
        method: str = "token", 
        **kwargs
    ) -> List[ChunkResult]:
        """
        批量文档分块方法
        
        Args:
            documents: 文档字典，{doc_id: content}
            method: 分块方法，"token" 或 "separator"
            **kwargs: 其他参数传递给分块方法
            
        Returns:
            分块结果列表
        """
        doc_ids = list(documents.keys())
        texts = list(documents.values())
        
        if method == "token":
            return self.chunking_by_token_size(texts, doc_ids, **kwargs)
        elif method == "separator":
            return self.chunking_by_separator(texts, doc_ids, **kwargs)
        else:
            raise ValueError(f"不支持的分块方法: {method}. 请使用 'token' 或 'separator'")

    def export_chunks_to_dict(self, chunks: List[ChunkResult]) -> Dict[str, Dict]:
        """
        将分块结果导出为字典格式（与GraphRAG兼容）
        
        Args:
            chunks: 分块结果列表
            
        Returns:
            字典格式的分块数据
        """
        result = {}
        for chunk in chunks:
            result[chunk.hash_id] = {
                "tokens": chunk.token_count,
                "content": chunk.content,
                "chunk_order_index": chunk.chunk_index,
                "full_doc_id": chunk.doc_id,
            }
        return result

    def save_chunks_to_file(self, chunks: List[ChunkResult], filename: str):
        """
        将分块结果保存到JSON文件
        
        Args:
            chunks: 分块结果列表
            filename: 输出文件名
        """
        import json
        
        data = self.export_chunks_to_dict(chunks)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"分块结果已保存到: {filename}")


def demo():
    """演示文档分块功能"""
    print("=== 文档分块功能演示 ===\n")
    
    # 示例文本
    sample_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

    机器学习是人工智能的一个重要分支。通过算法使机器能够从数据中学习并改善性能，无需进行明确的编程。深度学习又是机器学习的一个分支，它基于人工神经网络进行学习。

    近年来，随着计算能力的提升和大数据的发展，人工智能技术取得了突破性进展，在图像识别、语音识别、自然语言处理等领域达到了前所未有的水平。
    """
    
    # 创建分块器
    chunker = DocumentChunker(
        default_chunk_size=200,  # 使用较小的分块以便演示
        default_overlap_size=50
    )
    
    print("1. 基于Token数量的分块:")
    print("-" * 50)
    token_chunks = chunker.chunk_text(sample_text, method="token")
    
    for i, chunk in enumerate(token_chunks):
        print(f"分块 {i+1}:")
        print(f"  Token数量: {chunk.token_count}")
        print(f"  内容: {chunk.content[:100]}...")
        print(f"  哈希ID: {chunk.hash_id}")
        print()
    
    print("\n2. 基于分隔符的分块:")
    print("-" * 50)
    separator_chunks = chunker.chunk_text(sample_text, method="separator")
    
    for i, chunk in enumerate(separator_chunks):
        print(f"分块 {i+1}:")
        print(f"  Token数量: {chunk.token_count}")
        print(f"  内容: {chunk.content[:100]}...")
        print(f"  哈希ID: {chunk.hash_id}")
        print()
    
    # 保存到文件
    chunker.save_chunks_to_file(token_chunks, "demo_chunks.json")
    print("演示完成！")


if __name__ == "__main__":
    demo() 