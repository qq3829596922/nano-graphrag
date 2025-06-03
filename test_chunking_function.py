#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 chunking_by_token_size 函数的独立测试文件
"""

import tiktoken
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nano_graphrag._op import chunking_by_token_size


def test_chunking_by_token_size():
    """
    测试 chunking_by_token_size 函数的功能
    """
    print("开始测试 chunking_by_token_size 函数...")
    
    # 准备测试数据
    # 创建tiktoken编码器
    encoder = tiktoken.encoding_for_model("gpt-4o")
    
    # 测试文档内容
    test_docs = [
        "这是第一个测试文档。它包含了一些中文内容用于测试分块功能。我们需要确保分块算法能够正确处理不同长度的文本，并且能够保持适当的重叠。",
        "This is the second test document. It contains English content to test the chunking functionality. We need to ensure that the chunking algorithm can handle texts of different lengths and maintain proper overlap between chunks.",
        "短文档"
    ]
    
    # 文档键
    doc_keys = ["doc_1", "doc_2", "doc_3"]
    
    # 将文档转换为token列表
    tokens_list = []
    for doc in test_docs:
        tokens = encoder.encode(doc)
        tokens_list.append(tokens)
        print(f"文档: '{doc[:30]}...' -> Token数量: {len(tokens)}")
    
    print(f"\n原始tokens_list长度: {len(tokens_list)}")
    print(f"文档键: {doc_keys}")
    
    # 测试参数
    overlap_token_size = 20
    max_token_size = 50
    
    print(f"\n测试参数:")
    print(f"重叠token大小: {overlap_token_size}")
    print(f"最大token大小: {max_token_size}")
    
    # 调用函数进行测试
    result = chunking_by_token_size(
        tokens_list=tokens_list,
        doc_keys=doc_keys,
        tiktoken_model=encoder,
        overlap_token_size=overlap_token_size,
        max_token_size=max_token_size
    )
    
    # 输出结果
    print(f"\n测试结果:")
    print(f"总分块数量: {len(result)}")
    print("\n分块详情:")
    for i, chunk in enumerate(result):
        print(f"分块 {i+1}:")
        print(f"  - 文档ID: {chunk['full_doc_id']}")
        print(f"  - 分块索引: {chunk['chunk_order_index']}")
        print(f"  - Token数量: {chunk['tokens']}")
        print(f"  - 内容: '{chunk['content'][:50]}...'")
        print()
    
    # 验证测试结果
    print("验证测试结果:")
    
    # 检查返回值结构
    assert isinstance(result, list), "返回值应该是列表"
    assert len(result) > 0, "应该有分块结果"
    
    # 检查每个分块的结构
    for chunk in result:
        assert isinstance(chunk, dict), "每个分块应该是字典"
        assert "tokens" in chunk, "分块应该包含tokens字段"
        assert "content" in chunk, "分块应该包含content字段"
        assert "chunk_order_index" in chunk, "分块应该包含chunk_order_index字段"
        assert "full_doc_id" in chunk, "分块应该包含full_doc_id字段"
        assert chunk["full_doc_id"] in doc_keys, "文档ID应该在原始文档键中"
        assert chunk["tokens"] <= max_token_size, "分块token数量不应超过最大限制"
    
    # 检查文档覆盖情况
    doc_ids_in_result = set(chunk["full_doc_id"] for chunk in result)
    assert doc_ids_in_result == set(doc_keys), "所有文档都应该被处理"
    
    print("✅ 所有测试通过！")
    print(f"✅ 成功处理了 {len(doc_keys)} 个文档，生成了 {len(result)} 个分块")
    
    # 详细分析函数行为
    print("\n🔍 函数行为分析:")
    
    # 分析每个文档的分块情况
    for doc_key in doc_keys:
        doc_chunks = [chunk for chunk in result if chunk["full_doc_id"] == doc_key]
        print(f"\n文档 {doc_key}:")
        print(f"  - 原文档token数: {len(tokens_list[doc_keys.index(doc_key)])}")
        print(f"  - 生成分块数: {len(doc_chunks)}")
        
        for chunk in doc_chunks:
            print(f"    分块 {chunk['chunk_order_index']}: {chunk['tokens']} tokens")
    
    # 分析重叠情况
    print(f"\n📊 重叠情况分析:")
    print(f"当文档长度超过 {max_token_size} tokens时，会按照滑动窗口进行分块")
    print(f"每个分块最大 {max_token_size} tokens，重叠 {overlap_token_size} tokens")
    print(f"滑动步长: {max_token_size - overlap_token_size} tokens")
    
    return result


if __name__ == "__main__":
    test_result = test_chunking_by_token_size() 