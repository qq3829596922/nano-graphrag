#!/usr/bin/env python3
"""
第一步：文档处理模块
目标：实现文档的输入、ID生成、存储到JSON

您要实现的功能：
1. 文档ID生成
2. 文档存储到JSON文件
3. 文档加载和查看
4. 重复文档检测
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict
from datetime import datetime

class DocumentProcessor:
    """文档处理器"""
    def __init__(self, storage_dir: str = "./step1_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.docs_file = self.storage_dir / "documents.json"
        
        # 加载已有文档
        self.documents = self._load_documents()
    
    def _load_documents(self) -> Dict:
        """加载已存储的文档"""
        if self.docs_file.exists():
            with open(self.docs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_documents(self):
        """保存文档到JSON文件"""
        with open(self.docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def generate_doc_id(self, content: str) -> str:
        """
        生成文档ID
        
        TODO: 请实现这个方法
        要求：
        1. 使用MD5哈希算法
        2. 格式：doc-{hash前8位}
        3. 确保相同内容生成相同ID
        
        提示：
        - 使用 hashlib.md5()
        - 记得 encode('utf-8')
        - 使用 .hexdigest()[:8]
        """
        # 在这里实现您的代码
        pass
    
    def add_document(self, content: str, metadata: Dict = None) -> str:
        """
        添加文档
        
        TODO: 请实现这个方法
        要求：
        1. 生成文档ID
        2. 检查是否重复（如果ID已存在，跳过）
        3. 存储文档信息
        4. 保存到JSON文件
        
        返回：文档ID
        """
        doc_id = self.generate_doc_id(content)
        
        # 检查重复
        if doc_id in self.documents:
            print(f"⚠️  文档已存在: {doc_id}")
            return doc_id
        
        # TODO: 创建文档对象并存储
        # 文档对象应该包含：
        # - content: 文档内容
        # - metadata: 元数据（可选）
        # - created_at: 创建时间
        # - doc_id: 文档ID
        
        # TODO: 保存到文件
        
        print(f"✅ 已添加文档: {doc_id}")
        return doc_id
    
    def get_document(self, doc_id: str) -> Dict:
        """获取文档"""
        return self.documents.get(doc_id)
    
    def list_documents(self):
        """列出所有文档"""
        print(f"📚 共有 {len(self.documents)} 个文档:")
        for doc_id, doc_data in self.documents.items():
            preview = doc_data['content'][:50] + "..." if len(doc_data['content']) > 50 else doc_data['content']
            created_at = doc_data.get('created_at', '未知')
            print(f"  📄 {doc_id}: {preview} (创建时间: {created_at})")
    
    def get_stats(self):
        """获取统计信息"""
        if not self.documents:
            print("📊 统计信息: 暂无文档")
            return
        
        total_docs = len(self.documents)
        total_chars = sum(len(doc['content']) for doc in self.documents.values())
        avg_length = total_chars // total_docs
        
        print(f"📊 统计信息:")
        print(f"  文档数量: {total_docs}")
        print(f"  总字符数: {total_chars}")
        print(f"  平均长度: {avg_length} 字符")

def test_document_processor():
    """测试文档处理器"""
    print("🧪 测试文档处理器")
    print("=" * 40)
    
    # 创建处理器
    processor = DocumentProcessor()
    
    # 测试文档
    test_docs = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，它使计算机能够在不被明确编程的情况下学习。",
        "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。",
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"  # 重复文档
    ]
    
    # 添加文档
    print("\n📝 添加文档:")
    for i, doc in enumerate(test_docs):
        doc_id = processor.add_document(doc, {"source": f"test_doc_{i}"})
    
    print("\n📋 文档列表:")
    processor.list_documents()
    
    print("\n📊 统计信息:")
    processor.get_stats()
    
    print("\n🔍 测试文档检索:")
    # 获取第一个文档
    all_docs = list(processor.documents.keys())
    if all_docs:
        first_doc = processor.get_document(all_docs[0])
        print(f"文档详情: {first_doc}")

if __name__ == "__main__":
    test_document_processor()
    
    print("\n" + "="*50)
    print("🎯 完成第一步后，请回答以下问题：")
    print("1. 为什么要使用哈希生成ID而不是自增数字？")
    print("2. JSON存储的优缺点是什么？")
    print("3. 如何处理文档更新的情况？")
    print("4. 您觉得文档对象还应该包含哪些字段？")
    print("完成后，我们将进入第二步：文本分块") 