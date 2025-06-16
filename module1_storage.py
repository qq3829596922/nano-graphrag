#!/usr/bin/env python3
"""
📚 Module 1: 存储系统学习
学习GraphRAG如何组织和存储数据

学习目标：
1. 理解不同类型的存储（KV存储、向量数据库、图存储）
2. 观察数据的实际存储格式
3. 理解命名空间的概念
4. 学会手动操作存储系统

前置条件：无
预计时间：30分钟
"""

import asyncio
import json
from pathlib import Path
from nano_graphrag._storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage

class StorageExplorer:
    """存储系统探索器"""
    
    def __init__(self):
        self.working_dir = "./storage_demo"
        Path(self.working_dir).mkdir(exist_ok=True)
        
    async def demo_kv_storage(self):
        """演示键值存储"""
        print("🔑 演示 1：键值存储 (JsonKVStorage)")
        print("=" * 40)
        
        # 创建存储实例
        storage = JsonKVStorage(
            namespace="demo_docs",
            global_config={"working_dir": self.working_dir}
        )
        
        print("📝 插入一些示例数据...")
        test_data = {
            "doc1": {"content": "这是第一篇文档", "author": "张三"},
            "doc2": {"content": "这是第二篇文档", "author": "李四"},
            "doc3": {"content": "这是第三篇文档", "author": "王五"}
        }
        
        await storage.upsert(test_data)
        print(f"✅ 已插入 {len(test_data)} 条记录")
        
        print("\n🔍 检索数据...")
        # 获取单个记录
        doc1 = await storage.get_by_id("doc1")
        print(f"获取 doc1: {doc1}")
        
        # 获取多个记录
        docs = await storage.get_by_ids(["doc1", "doc2"])
        print(f"获取多个文档: {len(docs)} 条")
        
        print("\n📁 检查存储文件...")
        storage_file = Path(self.working_dir) / "demo_docs.json"
        if storage_file.exists():
            print(f"存储文件位置: {storage_file}")
            with open(storage_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            print(f"文件内容预览: {list(content.keys())}")
        
        print("\n🤔 思考题：")
        print("1. KV存储适合存储什么类型的数据？")
        print("2. 为什么使用JSON格式存储？")
        print("3. namespace参数的作用是什么？")
        input("按回车继续...")
        
    async def demo_vector_storage(self):
        """演示向量存储"""
        print("\n🎯 演示 2：向量存储 (NanoVectorDBStorage)")
        print("=" * 40)
        
        # 简单的嵌入函数（实际中会使用OpenAI等API）
        async def simple_embedding(texts):
            """简单的嵌入函数示例"""
            import random
            return [[random.random() for _ in range(4)] for _ in texts]
        
        # 创建向量存储
        vector_storage = NanoVectorDBStorage(
            namespace="demo_vectors",
            global_config={"working_dir": self.working_dir},
            embedding_func=simple_embedding
        )
        
        print("📝 插入向量数据...")
        vector_data = {
            "text1": {"content": "人工智能是未来"},
            "text2": {"content": "机器学习很有趣"},  
            "text3": {"content": "深度学习效果好"}
        }
        
        await vector_storage.upsert(vector_data)
        print(f"✅ 已插入 {len(vector_data)} 个向量")
        
        print("\n🔍 相似性搜索...")
        # 搜索相似内容
        results = await vector_storage.query("人工智能", top_k=2)
        print("搜索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. ID: {result['id']}, 相似度: {result.get('distance', 'N/A')}")
        
        print("\n📁 检查向量存储文件...")
        vector_dir = Path(self.working_dir) / "demo_vectors"
        if vector_dir.exists():
            print(f"向量存储目录: {vector_dir}")
            print("目录内容:")
            for file in vector_dir.iterdir():
                print(f"  📄 {file.name}")
        
        print("\n🤔 思考题：")
        print("1. 向量存储和KV存储有什么区别？")
        print("2. 为什么需要嵌入函数？")
        print("3. 相似性搜索是如何工作的？")
        input("按回车继续...")
        
    async def demo_graph_storage(self):
        """演示图存储"""
        print("\n🕸️ 演示 3：图存储 (NetworkXStorage)")
        print("=" * 40)
        
        # 创建图存储
        graph_storage = NetworkXStorage(
            namespace="demo_graph",
            global_config={"working_dir": self.working_dir}
        )
        
        print("📝 添加实体和关系...")
        
        # 添加实体
        entities = [
            {"entity_name": "苹果公司", "entity_type": "组织"},
            {"entity_name": "iPhone", "entity_type": "产品"},
            {"entity_name": "乔布斯", "entity_type": "人物"}
        ]
        
        for entity in entities:
            await graph_storage.upsert_node(
                entity["entity_name"], 
                node_data=entity
            )
        
        # 添加关系
        relations = [
            ("苹果公司", "开发", "iPhone"),
            ("乔布斯", "创立", "苹果公司"),
            ("乔布斯", "发布", "iPhone")
        ]
        
        for source, relation, target in relations:
            await graph_storage.upsert_edge(
                source, target,
                edge_data={"relation": relation}
            )
        
        print(f"✅ 已添加 {len(entities)} 个实体和 {len(relations)} 个关系")
        
        print("\n🔍 图分析...")
        # 获取节点信息
        apple_node = await graph_storage.get_node("苹果公司")
        print(f"苹果公司节点: {apple_node}")
        
        # 获取邻居
        neighbors = await graph_storage.get_node_edges("苹果公司")
        print(f"苹果公司的关系: {len(neighbors)} 个")
        
        print("\n📁 检查图存储文件...")
        graph_file = Path(self.working_dir) / "demo_graph.graphml"
        if graph_file.exists():
            print(f"图文件位置: {graph_file}")
            print(f"文件大小: {graph_file.stat().st_size} 字节")
        
        print("\n🤔 思考题：")
        print("1. 图存储适合表示什么类型的数据？")
        print("2. 节点和边分别存储什么信息？")
        print("3. GraphML格式是什么？")
        input("按回车继续...")
        
    def explore_storage_files(self):
        """探索生成的存储文件"""
        print("\n📁 存储文件总览")
        print("=" * 40)
        
        working_dir = Path(self.working_dir)
        if not working_dir.exists():
            print("没有找到存储文件")
            return
            
        print("生成的文件和目录：")
        for item in working_dir.iterdir():
            if item.is_file():
                size = item.stat().st_size
                print(f"📄 {item.name} ({size} 字节)")
            else:
                file_count = len(list(item.iterdir()))
                print(f"📁 {item.name}/ ({file_count} 个文件)")
        
        print("\n💡 动手实验：")
        print("1. 用文本编辑器打开 .json 文件，观察数据格式")
        print("2. 探索向量存储目录中的文件")
        print("3. 用图形工具打开 .graphml 文件（如果有的话）")
        
    async def cleanup_demo(self):
        """清理演示文件"""
        import shutil
        response = input("\n🗑️ 是否删除演示文件？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(self.working_dir, ignore_errors=True)
            print("✅ 演示文件已清理")

async def main():
    """主函数"""
    print("📚 Module 1: 存储系统学习")
    print("学习GraphRAG的数据持久化机制")
    print("=" * 50)
    
    explorer = StorageExplorer()
    
    try:
        await explorer.demo_kv_storage()
        await explorer.demo_vector_storage()  
        await explorer.demo_graph_storage()
        explorer.explore_storage_files()
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("请检查是否安装了所需依赖")
    finally:
        await explorer.cleanup_demo()
    
    print("\n🎯 学习小结：")
    print("1. KV存储：简单的键值对，适合文档和配置")
    print("2. 向量存储：支持相似性搜索，适合语义检索")
    print("3. 图存储：表示实体关系，适合知识图谱")
    print("\n✅ Module 1 完成！准备学习 Module 2: 文本处理")

if __name__ == "__main__":
    asyncio.run(main()) 