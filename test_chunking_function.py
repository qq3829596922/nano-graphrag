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
        
   
        # 显示实体列表

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
        
        matrix_array = np.frombuffer(matrix_bytes, dtype=np.float32)

# # 3. 重塑为正确的矩阵形状 (13个实体 x 1536维)
#         matrix = matrix_array.reshape(13, 1536)

        print(matrix_array)

        with open("matrix_array.txt", "w", encoding="utf-8") as f:
            f.write(str(matrix_array))
        # 转换为numpy数组
    except Exception as e:
        print(f"❌ 解码失败: {e}")
        return
        
        # 重塑为矩阵形状
      





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
    print(result)


if __name__ == "__main__":
    main()