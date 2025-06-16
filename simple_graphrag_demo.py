#!/usr/bin/env python3
"""
GraphRAG 简化版演示
逐步展示GraphRAG的核心工作流程
学习路径：
1. 运行这个脚本看整体效果
2. 逐个注释掉步骤，理解每一步的作用
3. 修改参数，观察结果变化
4. 深入研究每个模块的实现
"""

import asyncio
from nano_graphrag import GraphRAG, QueryParam

from dotenv import load_dotenv
import os


load_dotenv(override=True)

async def demo_step_by_step():
    """
    分步骤演示GraphRAG的工作流程
    
    """
    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("OPENAI_BASE_URL"))
    print("🚀 开始GraphRAG演示")
    print("=" * 50)
    
    # 第1步：创建最简配置的GraphRAG实例
    print("📋 第1步：初始化GraphRAG")
    rag = GraphRAG(
        working_dir="./demo_cache",  # 只设置工作目录，其他用默认值
        enable_llm_cache=True,       # 启用缓存节省API调用
    )
    print(f"✅ 工作目录: {rag.working_dir}")
    print()
    
    # 第2步：准备测试文档
    print("📄 第2步：准备测试文档")
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，它使计算机能够在不被明确编程的情况下学习。",
        "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。",
        "OpenAI是一家专注于人工智能研究的公司，开发了GPT系列模型。",
        "ChatGPT是OpenAI开发的对话AI模型，基于GPT架构构建。"
    ]
    
    for i, doc in enumerate(documents, 1):
        print(f"   文档{i}: {doc[:30]}...")
    print()
    
    # 第3步：插入文档（这里会执行完整的处理流程）
    print("⚙️ 第3步：处理文档（文本分块 → 实体提取 → 图构建 → 社区检测）")
    print("这一步可能需要几分钟，请耐心等待...")
    
    try:
        await rag.ainsert(documents)
        print("✅ 文档处理完成！")
    except Exception as e:
        print(f"❌ 处理出错: {e}")
        return
    print()
    
    # 第4步：测试三种不同的查询模式
    query = "什么是人工智能？"
    print(f"❓ 第4步：测试查询 - '{query}'")
    print("-" * 30)
    
    # 模式1: 简单RAG（基于文本块相似度）
    print("🔍 模式1: 简单RAG查询")
    try:
        result_naive = await rag.aquery(query, QueryParam(mode="naive"))
        print(f"回答: {result_naive}")
    except Exception as e:
        print(f"❌ 简单RAG查询出错: {e}")
    print()
    
    # 模式2: 本地查询（基于实体和社区）
    print("🎯 模式2: 本地查询")
    try:
        result_local = await rag.aquery(query, QueryParam(mode="local"))
        print(f"回答: {result_local}")
    except Exception as e:
        print(f"❌ 本地查询出错: {e}")
    print()
    
    # 模式3: 全局查询（基于所有社区报告）
    print("🌍 模式3: 全局查询")
    try:
        result_global = await rag.aquery(query, QueryParam(mode="global"))
        print(f"回答: {result_global}")
    except Exception as e:
        print(f"❌ 全局查询出错: {e}")
    print()
    
    print("🎉 演示完成！")
    print("=" * 50)

def main():
    """主函数"""
    print("GraphRAG 学习演示")
    print("注意：首次运行需要API密钥，请确保已配置OpenAI API Key")
    print()
    
    # 运行异步演示
    asyncio.run(demo_step_by_step())

if __name__ == "__main__":
    main() 