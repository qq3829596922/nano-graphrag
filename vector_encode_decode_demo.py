#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量编码解码演示脚本
体验3个100维向量的encode和decode过程
"""

import numpy as np
import base64
import json
from typing import List


def create_sample_vectors() -> np.ndarray:
    """
    创建3个100维的示例向量
    
    Returns:
        np.ndarray: 形状为(3, 100)的向量矩阵
    """
    print("🎲 正在生成3个100维向量...")
    
    # 设置随机种子，确保结果可重现
    np.random.seed(42)
    
    # 方法1：完全随机向量
    vector1 = np.random.randn(100).astype(np.float32)
    
    # 方法2：带模式的向量（前50维为正，后50维为负）
    vector2 = np.concatenate([
        np.random.rand(50) + 1,    # 正值区间[1, 2]
        -(np.random.rand(50) + 1)  # 负值区间[-2, -1]
    ]).astype(np.float32)
    
    # 方法3：周期性向量
    x = np.linspace(0, 4*np.pi, 100)
    vector3 = (np.sin(x) + 0.5 * np.cos(2*x) + 0.1 * np.random.randn(100)).astype(np.float32)
    
    # 组合成矩阵
    vectors = np.vstack([vector1, vector2, vector3])
    
    # 归一化向量（可选）
    # vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    return vectors


def display_vectors(vectors: np.ndarray, title: str):
    """
    显示向量的基本信息
    
    Args:
        vectors (np.ndarray): 向量矩阵
        title (str): 标题
    """
    print(f"\n{title}")
    print("=" * 50)
    
    for i, vector in enumerate(vectors):
        print(f"📊 向量 {i+1}:")
        print(f"   形状: {vector.shape}")
        print(f"   数据类型: {vector.dtype}")
        print(f"   范数: {np.linalg.norm(vector):.6f}")
        print(f"   最大值: {vector.max():.6f}")
        print(f"   最小值: {vector.min():.6f}")
        print(f"   平均值: {vector.mean():.6f}")
        print(f"   前5维: [{', '.join(f'{x:.4f}' for x in vector[:5])}]")
        print(f"   后5维: [{', '.join(f'{x:.4f}' for x in vector[-5:])}]")
        print()


def encode_vectors(vectors: np.ndarray) -> str:
    """
    将向量矩阵编码为Base64字符串
    
    Args:
        vectors (np.ndarray): 向量矩阵
        
    Returns:
        str: Base64编码的字符串
    """
    print("🔒 正在编码向量...")
    
    # 将numpy数组转换为字节
    vector_bytes = vectors.tobytes()
    
    print(f"   原始数据大小: {len(vector_bytes)} 字节")
    print(f"   数据类型: {vectors.dtype}")
    print(f"   数组形状: {vectors.shape}")
    
    # Base64编码
    encoded_string = base64.b64encode(vector_bytes).decode('utf-8')
    
    print(f"   编码后长度: {len(encoded_string)} 字符")
    print(f"   压缩比例: {len(encoded_string)/len(vector_bytes):.2f}")
    
    return encoded_string


def decode_vectors(encoded_string: str, shape: tuple, dtype: str = 'float32') -> np.ndarray:
    """
    将Base64字符串解码回向量矩阵
    
    Args:
        encoded_string (str): Base64编码的字符串
        shape (tuple): 原始矩阵形状
        dtype (str): 数据类型
        
    Returns:
        np.ndarray: 解码后的向量矩阵
    """
    print("🔓 正在解码向量...")
    
    # Base64解码
    vector_bytes = base64.b64decode(encoded_string.encode('utf-8'))
    
    print(f"   解码后数据大小: {len(vector_bytes)} 字节")
    
    # 转换回numpy数组
    vectors = np.frombuffer(vector_bytes, dtype=dtype).reshape(shape)
    
    print(f"   恢复后形状: {vectors.shape}")
    print(f"   恢复后类型: {vectors.dtype}")
    
    return vectors


def verify_integrity(original: np.ndarray, decoded: np.ndarray) -> bool:
    """
    验证编码解码的完整性
    
    Args:
        original (np.ndarray): 原始向量
        decoded (np.ndarray): 解码后的向量
        
    Returns:
        bool: 是否完全一致
    """
    print("🔍 验证数据完整性...")
    
    # 检查形状
    shape_match = original.shape == decoded.shape
    print(f"   形状匹配: {shape_match} ({original.shape} vs {decoded.shape})")
    
    # 检查数据类型
    dtype_match = original.dtype == decoded.dtype
    print(f"   类型匹配: {dtype_match} ({original.dtype} vs {decoded.dtype})")
    
    # 检查数值
    values_match = np.allclose(original, decoded, rtol=1e-7, atol=1e-7)
    print(f"   数值匹配: {values_match}")
    
    if values_match:
        max_diff = np.max(np.abs(original - decoded))
        print(f"   最大差异: {max_diff:.2e}")
    else:
        differences = np.abs(original - decoded)
        print(f"   差异统计:")
        print(f"     最大差异: {np.max(differences):.6f}")
        print(f"     平均差异: {np.mean(differences):.6f}")
        print(f"     差异数量: {np.sum(differences > 1e-6)}")
    
    return shape_match and dtype_match and values_match


def save_demo_results(original_vectors: np.ndarray, encoded_string: str, decoded_vectors: np.ndarray):
    """
    保存演示结果到文件
    
    Args:
        original_vectors (np.ndarray): 原始向量
        encoded_string (str): 编码字符串
        decoded_vectors (np.ndarray): 解码向量
    """
    print("💾 保存演示结果...")
    
    # 创建演示数据
    demo_data = {
        "info": {
            "description": "向量编码解码演示",
            "vector_count": original_vectors.shape[0],
            "vector_dimension": original_vectors.shape[1],
            "data_type": str(original_vectors.dtype)
        },
        "original_vectors": {
            "shape": original_vectors.shape,
            "data": original_vectors.tolist()
        },
        "encoded_base64": encoded_string,
        "decoded_vectors": {
            "shape": decoded_vectors.shape,
            "data": decoded_vectors.tolist()
        }
    }
    
    # 保存为JSON
    with open("vector_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    
    print("   ✅ 结果已保存到: vector_demo_results.json")
    
    # 保存编码字符串到单独文件
    with open("encoded_vectors.txt", "w", encoding="utf-8") as f:
        f.write("# Base64编码的向量数据\n")
        f.write(f"# 向量数量: {original_vectors.shape[0]}\n")
        f.write(f"# 向量维度: {original_vectors.shape[1]}\n")
        f.write(f"# 数据类型: {original_vectors.dtype}\n")
        f.write(f"# 编码长度: {len(encoded_string)} 字符\n")
        f.write("\n")
        f.write(encoded_string)
    
    print("   ✅ 编码数据已保存到: encoded_vectors.txt")


def main():
    """
    主函数：完整的编码解码演示流程
    """
    print("🚀 向量编码解码演示开始！")
    print("=" * 60)
    
    # 步骤1：创建示例向量
    original_vectors = create_sample_vectors()
    display_vectors(original_vectors, "📋 原始向量信息")
    
    # 步骤2：编码向量
    print("\n" + "="*60)
    encoded_string = encode_vectors(original_vectors)
    
    # 显示编码结果的一部分
    print(f"\n🔒 编码结果预览:")
    print(f"   开头: {encoded_string[:50]}...")
    print(f"   结尾: ...{encoded_string[-50:]}")
    print(f"   完整长度: {len(encoded_string)} 字符")
    
    # 步骤3：解码向量
    print("\n" + "="*60)
    decoded_vectors = decode_vectors(
        encoded_string, 
        original_vectors.shape, 
        str(original_vectors.dtype)
    )
    
    # 步骤4：显示解码结果
    display_vectors(decoded_vectors, "📋 解码后向量信息")
    
    # 步骤5：验证完整性
    print("\n" + "="*60)
    is_valid = verify_integrity(original_vectors, decoded_vectors)
    
    if is_valid:
        print("✅ 编码解码过程完全成功！数据完整性100%保持！")
    else:
        print("❌ 检测到数据不一致，请检查编码解码过程")
    
    # 步骤6：保存结果
    print("\n" + "="*60)
    save_demo_results(original_vectors, encoded_string, decoded_vectors)
    
    print(f"\n🎉 演示完成！")
    print(f"   你现在理解了向量是如何被编码为Base64字符串，")
    print(f"   然后又如何被完美地解码回原始向量的！")


if __name__ == "__main__":
    main() 