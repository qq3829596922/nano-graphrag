#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据解码脚本
用于解码 nano-vectordb 存储的向量数据
"""

import json
import base64
import numpy as np
import os


def decode_vector_database(json_file_path):
    """
    解码向量数据库文件
    
    Args:
        json_file_path (str): JSON文件路径
    """
    print(f"🔍 正在读取文件: {json_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ 文件不存在: {json_file_path}")
        return
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("✅ JSON文件读取成功!")
        
        # 显示基本信息
        print(f"\n📊 基本信息:")
        print(f"   向量维度: {data.get('embedding_dim', 'Unknown')}")
        print(f"   实体数量: {len(data.get('data', []))}")
        
        # 显示实体列表
        print(f"\n👥 实体列表:")
        entities = data.get('data', [])
        for i, entity in enumerate(entities):
            entity_id = entity.get('__id__', 'Unknown')
            entity_name = entity.get('entity_name', 'Unknown')
            print(f"   {i+1:2d}. {entity_name} (ID: {entity_id[:20]}...)")
        
        # 解码向量矩阵
        matrix_encoded = data.get('matrix', '')
        if not matrix_encoded:
            print("❌ 未找到向量数据")
            return
        
        print(f"\n🔓 正在解码向量数据...")
        print(f"   编码数据长度: {len(matrix_encoded)} 字符")
        
        # Base64解码
        matrix_bytes = base64.b64decode(matrix_encoded)
        print(f"   二进制数据长度: {len(matrix_bytes)} 字节")
        
        # 转换为numpy数组
        # nano-vectordb 使用 float32 格式存储
        matrix_array = np.frombuffer(matrix_bytes, dtype=np.float32)
        
        # 重塑为矩阵形状
        embedding_dim = data.get('embedding_dim', 1536)
        num_entities = len(entities)
        
        if len(matrix_array) != num_entities * embedding_dim:
            print(f"⚠️  数据长度不匹配:")
            print(f"   期望: {num_entities} × {embedding_dim} = {num_entities * embedding_dim}")
            print(f"   实际: {len(matrix_array)}")
        
        # 重塑矩阵
        vectors = matrix_array.reshape(num_entities, embedding_dim)
        
        print(f"✅ 向量解码成功!")
        print(f"   矩阵形状: {vectors.shape}")
        print(f"   数据类型: {vectors.dtype}")
        
        # 显示统计信息
        print(f"\n📈 向量统计信息:")
        print(f"   最小值: {vectors.min():.6f}")
        print(f"   最大值: {vectors.max():.6f}")
        print(f"   平均值: {vectors.mean():.6f}")
        print(f"   标准差: {vectors.std():.6f}")
        
        # 显示每个实体的向量信息
        print(f"\n🔢 实体向量详情:")
        for i, entity in enumerate(entities):
            entity_name = entity.get('entity_name', 'Unknown')
            vector = vectors[i]
            vector_norm = np.linalg.norm(vector)
            
            print(f"   {i+1:2d}. {entity_name}")
            print(f"       向量范数: {vector_norm:.6f}")
            print(f"       前10维度: [{', '.join(f'{x:.4f}' for x in vector[:10])}...]")
            print(f"       后10维度: [...{', '.join(f'{x:.4f}' for x in vector[-10:])}]")
            
            # 询问是否显示完整向量
            if i == 0:  # 只对第一个实体询问
                show_full = input(f"\n是否显示完整的1536维向量数据？(y/n): ").strip().lower()
                if show_full == 'y':
                    print(f"       完整向量: {vector.tolist()}")
            print()
        
        # 计算相似度示例
        print("🔍 相似度计算示例 (余弦相似度):")
        if len(vectors) >= 2:
            # 计算前两个实体的相似度
            v1, v2 = vectors[0], vectors[1]
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            name1 = entities[0].get('entity_name', 'Entity 1')
            name2 = entities[1].get('entity_name', 'Entity 2')
            
            print(f"   {name1} vs {name2}")
            print(f"   相似度: {similarity:.6f}")
        
        # 保存解码结果
        save_decoded_data(vectors, entities, json_file_path)
        
        return vectors, entities
        
    except Exception as e:
        print(f"❌ 解码失败: {str(e)}")
        import traceback
        traceback.print_exc()


def save_decoded_data(vectors, entities, original_file_path):
    """
    保存解码后的数据为可读格式
    
    Args:
        vectors (np.ndarray): 解码后的向量矩阵
        entities (list): 实体列表
        original_file_path (str): 原始文件路径
    """
    base_name = os.path.splitext(original_file_path)[0]
    
    # 保存为可读的JSON格式
    readable_data = {
        "metadata": {
            "total_entities": len(entities),
            "vector_dimension": vectors.shape[1],
            "data_type": str(vectors.dtype)
        },
        "entities_with_vectors": []
    }
    
    # 添加每个实体的向量数据
    for i, entity in enumerate(entities):
        entity_data = {
            "entity_name": entity.get('entity_name', ''),
            "entity_id": entity.get('__id__', ''),
            "vector": vectors[i].tolist(),  # 转换为Python列表
            "vector_norm": float(np.linalg.norm(vectors[i]))
        }
        readable_data["entities_with_vectors"].append(entity_data)
    
    # 保存为格式化的JSON文件
    json_file = f"{base_name}_decoded.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(readable_data, f, ensure_ascii=False, indent=2)
    print(f"💾 可读向量数据已保存到: {json_file}")
    
    # 保存为简化的文本格式
    txt_file = f"{base_name}_vectors.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("向量数据解码结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"实体数量: {len(entities)}\n")
        f.write(f"向量维度: {vectors.shape[1]}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, entity in enumerate(entities):
            entity_name = entity.get('entity_name', 'Unknown')
            vector = vectors[i]
            
            f.write(f"实体 {i+1}: {entity_name}\n")
            f.write(f"ID: {entity.get('__id__', '')}\n")
            f.write(f"向量范数: {np.linalg.norm(vector):.6f}\n")
            f.write(f"向量数据: {vector.tolist()}\n")
            f.write("-" * 30 + "\n\n")
    
    print(f"💾 文本格式数据已保存到: {txt_file}")


def analyze_similarity(vectors, entities, top_k=5):
    """
    分析实体间的相似度
    
    Args:
        vectors (np.ndarray): 向量矩阵
        entities (list): 实体列表
        top_k (int): 显示最相似的top_k个实体对
    """
    print(f"\n🔗 实体相似度分析 (Top {top_k}):")
    
    # 计算所有实体对的相似度
    similarities = []
    
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            v1, v2 = vectors[i], vectors[j]
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            name1 = entities[i].get('entity_name', f'Entity {i}')
            name2 = entities[j].get('entity_name', f'Entity {j}')
            
            similarities.append((similarity, name1, name2))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # 显示最相似的实体对
    for i, (sim, name1, name2) in enumerate(similarities[:top_k]):
        print(f"   {i+1:2d}. {name1} ↔ {name2}")
        print(f"       相似度: {sim:.6f}")


def main():
    """主函数"""
    print("🎯 向量数据解码工具")
    print("=" * 50)
    
    # 默认文件路径
    default_file = "mytest/vdb_entities.json"
    
    # 检查默认文件
    if os.path.exists(default_file):
        print(f"📁 找到默认文件: {default_file}")
        file_path = default_file
    else:
        # 让用户输入文件路径
        file_path = input("请输入 vdb_entities.json 文件路径: ").strip()
        if not file_path:
            print("❌ 未提供文件路径")
            return
    
    # 解码向量数据
    result = decode_vector_database(file_path)
    
    if result:
        vectors, entities = result
        
        # 进行相似度分析
        analyze_similarity(vectors, entities)
        
        print(f"\n🎉 解码完成！")
        print(f"   共解码 {len(entities)} 个实体的 {vectors.shape[1]} 维向量")


if __name__ == "__main__":
    main() 