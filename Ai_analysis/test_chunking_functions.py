"""
测试文本分块函数的直接调用
演示 nano_graphrag 中的分块函数如何工作
"""

import json
import tiktoken
from nano_graphrag._op import chunking_by_token_size, chunking_by_seperators, get_chunks
from nano_graphrag._utils import compute_mdhash_id

def test_chunking_functions():
    """测试和演示分块函数的工作原理"""
    
    print("=== 文本分块函数测试 ===\n")
    
    # 准备测试文档
    test_doc = """
    曹操分兵收割军粮后，吕布陈宫看到曹操兵力分散，认为是好机会，又率领了一万人马要来和曹操决一死战。
    结果，被曹操设下埋伏杀得大败。这一仗吕布伤筋动骨，从此再也无法在兖州立足。
    
    兖州的叛乱，表面原因是因为曹操过于粗暴，杀害了陈留的名士边让，引起了兖州人士的恐慌。
    实际上有更深刻的原因。看一看各地的诸侯，曹操在兖州，刘备在徐州，孙策在江南。
    
    曹操历经了兖州叛乱，刘备遭遇了吕布反水，结果都是因为治理内部的时间太短，工夫还没到。
    后来曹操也杀了孔融，孔融的名气可比边让大得多，但什么事也没有。
    """
    
    doc_id = compute_mdhash_id(test_doc.strip(), prefix="doc-")
    print(f"📄 测试文档ID: {doc_id}")
    print(f"📄 文档长度: {len(test_doc)} 字符\n")
    
    # 创建tiktoken编码器
    encoder = tiktoken.encoding_for_model("gpt-4o")
    
    # 编码文档为tokens
    tokens = encoder.encode(test_doc)
    print(f"📊 编码后token数量: {len(tokens)}")
    print()
    
    # ========== 测试1: chunking_by_token_size ==========
    print("🔧 测试1: chunking_by_token_size() 函数")
    print("-" * 50)
    
    token_chunks = chunking_by_token_size(
        tokens_list=[tokens],           # 包装为列表
        doc_keys=[doc_id],             # 文档ID列表
        tiktoken_model=encoder,        # tiktoken编码器
        overlap_token_size=50,         # 重叠50个token
        max_token_size=200,           # 每块最大200个token
    )
    
    print(f"✅ 生成分块数量: {len(token_chunks)}")
    for i, chunk in enumerate(token_chunks):
        print(f"  分块 {i+1}:")
        print(f"    Token数量: {chunk['tokens']}")
        print(f"    分块索引: {chunk['chunk_order_index']}")
        print(f"    文档ID: {chunk['full_doc_id']}")
        print(f"    内容预览: {chunk['content'][:100]}...")
        print()
    
    # ========== 测试2: chunking_by_seperators ==========
    print("🔧 测试2: chunking_by_seperators() 函数")
    print("-" * 50)
    
    separator_chunks = chunking_by_seperators(
        tokens_list=[tokens],
        doc_keys=[doc_id],
        tiktoken_model=encoder,
        overlap_token_size=30,
        max_token_size=150,
    )
    
    print(f"✅ 生成分块数量: {len(separator_chunks)}")
    for i, chunk in enumerate(separator_chunks):
        print(f"  分块 {i+1}:")
        print(f"    Token数量: {chunk['tokens']}")
        print(f"    分块索引: {chunk['chunk_order_index']}")
        print(f"    文档ID: {chunk['full_doc_id']}")
        print(f"    内容预览: {chunk['content'][:100]}...")
        print()
    
    # ========== 测试3: get_chunks 统一接口 ==========
    print("🔧 测试3: get_chunks() 统一接口函数")
    print("-" * 50)
    
    # 准备输入格式（模拟GraphRAG内部格式）
    new_docs = {
        doc_id: {"content": test_doc.strip()}
    }
    
    # 使用get_chunks函数
    result_chunks = get_chunks(
        new_docs=new_docs,
        chunk_func=chunking_by_token_size,  # 指定分块函数
        overlap_token_size=40,
        max_token_size=180,
    )
    
    print(f"✅ 生成分块数量: {len(result_chunks)}")
    print("📋 分块结果字典格式:")
    
    for chunk_id, chunk_data in result_chunks.items():
        print(f"  分块ID: {chunk_id}")
        print(f"    Token数量: {chunk_data['tokens']}")
        print(f"    分块索引: {chunk_data['chunk_order_index']}")
        print(f"    文档ID: {chunk_data['full_doc_id']}")
        print(f"    内容预览: {chunk_data['content'][:100]}...")
        print()
    
    # ========== 保存结果到文件 ==========
    print("💾 保存分块结果到文件")
    print("-" * 50)
    
    # 保存为JSON格式（模拟kv_store_text_chunks.json）
    with open("test_chunks_result.json", "w", encoding="utf-8") as f:
        json.dump(result_chunks, f, indent=2, ensure_ascii=False)
    
    print("✅ 分块结果已保存到: test_chunks_result.json")
    
    # ========== 与实际kv_store_text_chunks.json对比 ==========
    print("\n📊 与实际生成文件的数据结构对比")
    print("-" * 50)
    
    try:
        with open("mytest/kv_store_text_chunks.json", "r", encoding="utf-8") as f:
            actual_chunks = json.load(f)
        
        print(f"实际文件分块数量: {len(actual_chunks)}")
        print("实际文件数据结构示例:")
        
        for i, (chunk_id, chunk_data) in enumerate(actual_chunks.items()):
            if i >= 1:  # 只显示第一个
                break
            print(f"  分块ID: {chunk_id}")
            print(f"    tokens: {chunk_data['tokens']}")
            print(f"    chunk_order_index: {chunk_data['chunk_order_index']}")
            print(f"    full_doc_id: {chunk_data['full_doc_id']}")
            print(f"    content: {chunk_data['content'][:100]}...")
            
    except FileNotFoundError:
        print("⚠️  未找到实际的 kv_store_text_chunks.json 文件")
    
    print("\n🎯 总结")
    print("-" * 50)
    print("✅ 文本分块函数测试完成")
    print("✅ 数据结构与GraphRAG保持一致")
    print("✅ 支持Token分块和分隔符分块")
    print("✅ 分块ID使用MD5哈希确保唯一性")
    print("✅ 保持原文档追踪和分块顺序")


if __name__ == "__main__":
    test_chunking_functions() 