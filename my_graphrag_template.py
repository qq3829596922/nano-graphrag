#!/usr/bin/env python3
"""
您的第一个GraphRAG实现
任务：填写TODO标记的部分，实现基本功能
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class MyGraphRAG:
    """
    您要实现的GraphRAG类
    
    第一课目标：实现基本的文档存储和检索
    """
    working_dir: str = "./my_graphrag_cache"
    
    def __post_init__(self):
        """初始化方法"""
        # TODO 1: 创建工作目录
        # 提示：使用 Path(self.working_dir).mkdir(exist_ok=True)
        pass
        
        # TODO 2: 初始化存储字典
        # 我们先用内存存储，后面再改成文件存储
        # 提示：需要存储文档、文本块、实体等
        pass
    
    def _compute_doc_id(self, content: str) -> str:
        """计算文档ID"""
        # TODO 3: 实现文档ID计算
        # 提示：使用MD5哈希，格式为 "doc-" + hash[:8]
        pass
    
    def _chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        """将文本分块"""
        # TODO 4: 实现简单的文本分块
        # 提示：按字符数分块，可以简单用切片
        pass
    
    async def insert(self, documents: List[str]):
        """插入文档"""
        print(f"📝 开始插入 {len(documents)} 个文档...")
        
        for doc in documents:
            # TODO 5: 为每个文档生成ID并分块
            doc_id = self._compute_doc_id(doc)
            chunks = self._chunk_text(doc)
            
            # TODO 6: 存储文档和块
            # 提示：保存到之前初始化的存储字典中
            pass
            
        print("✅ 文档插入完成")
    
    async def query(self, question: str) -> str:
        """查询"""
        print(f"🔍 查询: {question}")
        
        # TODO 7: 实现简单的关键词匹配查询
        # 提示：遍历所有文本块，找到包含关键词的块
        relevant_chunks = []
        
        # TODO 8: 返回相关文本块
        if relevant_chunks:
            return f"找到相关内容: {relevant_chunks[0][:100]}..."
        else:
            return "未找到相关内容"

# 测试代码
async def test_my_graphrag():
    """测试您的实现"""
    print("🧪 测试您的GraphRAG实现")
    print("=" * 40)
    
    # 创建实例
    rag = MyGraphRAG()
    
    # 测试文档
    docs = [
        "人工智能是计算机科学的一个分支，致力于创建智能机器。",
        "机器学习是人工智能的一个子领域，让计算机能够学习。",
        "深度学习使用神经网络来解决复杂问题。"
    ]
    
    # 插入文档
    await rag.insert(docs)
    
    # 测试查询
    result = await rag.query("什么是人工智能")
    print(f"查询结果: {result}")

if __name__ == "__main__":
    asyncio.run(test_my_graphrag()) 