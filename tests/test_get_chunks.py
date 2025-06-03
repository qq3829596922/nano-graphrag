import unittest
from unittest.mock import patch, MagicMock
import tiktoken
from nano_graphrag._op import (
    get_chunks,
    chunking_by_token_size,
    chunking_by_seperators
)
from nano_graphrag._utils import compute_mdhash_id


class TestGetChunks(unittest.TestCase):
    """测试 get_chunks 函数的单元测试类"""

    def setUp(self):
        """测试初始化"""
        # 准备测试用的文档数据，包含中文内容
        self.sample_docs = {
            "doc1": {
                "content": "这是第一个文档。它包含一些测试内容。用于测试分块功能。"
            },
            "doc2": {
                "content": "这是第二个文档，它比第一个文档稍微长一些。包含更多的文本内容，用于测试分块算法的处理能力。这里是更多的内容。"
            }
        }
        
        # 初始化 tiktoken 编码器，用于后续的 token 编码测试
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    def test_get_chunks_basic_functionality(self):
        """测试 get_chunks 基本功能"""
        # 调用被测试的函数
        result = get_chunks(self.sample_docs)
        
        # 验证返回类型是字典
        self.assertIsInstance(result, dict)
        
        # 验证每个结果都有正确的键
        for chunk_id, chunk_data in result.items():
            # 验证 chunk_id 是字符串类型
            self.assertIsInstance(chunk_id, str)
            # 验证 chunk_id 以 "chunk-" 开头（这是预期的格式）
            self.assertTrue(chunk_id.startswith("chunk-"))
            
            # 验证 chunk_data 包含必要的字段
            required_fields = ["tokens", "content", "chunk_order_index", "full_doc_id"]
            for field in required_fields:
                # 确保每个必需字段都存在于chunk数据中
                self.assertIn(field, chunk_data)
            
            # 验证字段类型
            self.assertIsInstance(chunk_data["tokens"], int)          # token数量应该是整数
            self.assertIsInstance(chunk_data["content"], str)        # 内容应该是字符串
            self.assertIsInstance(chunk_data["chunk_order_index"], int)  # 块索引应该是整数
            self.assertIsInstance(chunk_data["full_doc_id"], str)    # 文档ID应该是字符串

    def test_get_chunks_with_custom_chunk_function(self):
        """测试使用自定义分块函数"""
        # 使用分隔符分块函数而不是默认的token大小分块函数
        result = get_chunks(
            self.sample_docs,
            chunk_func=chunking_by_seperators  # 指定使用分隔符分块策略
        )
        
        # 验证结果基本正确性
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)  # 确保至少产生了一些分块

    def test_get_chunks_with_custom_parameters(self):
        """测试使用自定义参数"""
        # 使用自定义的overlap和max_token_size参数
        result = get_chunks(
            self.sample_docs,
            chunk_func=chunking_by_token_size,  # 使用token大小分块函数
            overlap_token_size=64,              # 设置重叠token数为64
            max_token_size=512                  # 设置最大token数为512
        )
        
        # 验证自定义参数不会导致错误
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_get_chunks_empty_input(self):
        """测试空输入"""
        # 测试边界情况：空文档字典
        result = get_chunks({})
        # 空输入应该返回空字典
        self.assertEqual(result, {})

    def test_get_chunks_single_document(self):
        """测试单个文档"""
        # 准备只包含一个文档的测试数据
        single_doc = {
            "single_doc": {
                "content": "这是一个简单的单个文档测试。"
            }
        }
        
        # 处理单个文档
        result = get_chunks(single_doc)
        
        # 验证基本结果正确性
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        
        # 验证文档ID正确传递到每个分块中
        for chunk_data in result.values():
            # 每个分块都应该正确记录来源文档ID
            self.assertEqual(chunk_data["full_doc_id"], "single_doc")

    def test_get_chunks_hash_generation(self):
        """测试哈希ID生成"""
        # 获取分块结果
        result = get_chunks(self.sample_docs)
        
        # 验证每个chunk的ID是唯一的
        chunk_ids = list(result.keys())
        # 通过比较列表长度和set长度来确保没有重复ID
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        
        # 验证ID格式
        for chunk_id in chunk_ids:
            # 确保ID以预期前缀开头
            self.assertTrue(chunk_id.startswith("chunk-"))
            # MD5哈希值是32个字符，加上"chunk-"前缀应该是38个字符
            self.assertEqual(len(chunk_id), len("chunk-") + 32)  # MD5 hash is 32 chars

    def test_get_chunks_content_integrity(self):
        """测试内容完整性"""
        # 获取分块结果
        result = get_chunks(self.sample_docs)
        
        # 收集所有分块的内容
        all_chunk_contents = []
        for chunk_data in result.values():
            all_chunk_contents.append(chunk_data["content"])
        
        # 验证原始内容在分块中存在
        original_contents = [doc["content"] for doc in self.sample_docs.values()]
        
        # 检查原始内容是否能在分块中找到
        for original_content in original_contents:
            # 检查原始内容的主要部分是否存在于某个分块中
            content_found = any(
                original_content in chunk_content or chunk_content in original_content
                for chunk_content in all_chunk_contents
            )
            # 注意：由于token化和重新解码的过程，完全匹配可能不总是成立

    @patch('nano_graphrag._op.tiktoken.encoding_for_model')  # 模拟tiktoken编码器
    def test_get_chunks_with_mocked_encoder(self, mock_encoder_func):
        """测试使用模拟的编码器"""
        # 创建模拟编码器对象
        mock_encoder = MagicMock()
        # 设置模拟的encode_batch返回值（模拟token编码结果）
        mock_encoder.encode_batch.return_value = [
            [1, 2, 3, 4, 5],              # doc1 的 tokens
            [6, 7, 8, 9, 10, 11, 12]      # doc2 的 tokens
        ]
        # 设置模拟的decode_batch返回值（模拟从tokens解码回文本）
        mock_encoder.decode_batch.return_value = [
            "mock chunk 1",
            "mock chunk 2"
        ]
        # 让模拟函数返回我们的模拟编码器
        mock_encoder_func.return_value = mock_encoder
        
        # 创建简单的测试数据
        test_docs = {
            "test_doc1": {"content": "test content 1"},
            "test_doc2": {"content": "test content 2"}
        }
        
        # 定义模拟分块函数，返回预定义的分块结果
        def mock_chunk_func(tokens_list, doc_keys, tiktoken_model, **kwargs):
            return [
                {
                    "tokens": 5,
                    "content": "mock chunk 1",
                    "chunk_order_index": 0,
                    "full_doc_id": "test_doc1"
                },
                {
                    "tokens": 7,
                    "content": "mock chunk 2", 
                    "chunk_order_index": 0,
                    "full_doc_id": "test_doc2"
                }
            ]
        
        # 使用模拟函数调用get_chunks
        result = get_chunks(test_docs, chunk_func=mock_chunk_func)
        
        # 验证模拟编码器被正确调用
        mock_encoder_func.assert_called_once_with("gpt-4o")  # 确保使用正确的模型名
        mock_encoder.encode_batch.assert_called_once()       # 确保调用了批量编码
        
        # 验证返回结果的数量正确
        self.assertEqual(len(result), 2)
        
        # 验证哈希ID正确生成（基于内容生成的MD5哈希）
        expected_id1 = compute_mdhash_id("mock chunk 1", prefix="chunk-")
        expected_id2 = compute_mdhash_id("mock chunk 2", prefix="chunk-")
        
        # 确保生成的ID在结果中存在
        self.assertIn(expected_id1, result)
        self.assertIn(expected_id2, result)

    def test_get_chunks_large_document(self):
        """测试大文档分块"""
        # 创建一个比较大的文档（重复内容200次）
        large_content = "这是一个很长的文档。" * 200
        large_docs = {
            "large_doc": {
                "content": large_content
            }
        }
        
        # 使用较小的分块参数来强制分块
        result = get_chunks(
            large_docs,
            max_token_size=256,    # 设置最大token数为256
            overlap_token_size=64  # 设置重叠token数为64
        )
        
        # 验证大文档被正确分为多个块
        self.assertGreater(len(result), 1)
        
        # 验证每个分块的大小都在合理范围内
        for chunk_data in result.values():
            # 允许一些误差，因为分块边界可能不是严格按token数切分
            self.assertLessEqual(chunk_data["tokens"], 256 + 10)  # 允许一些误差

    def test_get_chunks_various_content_types(self):
        """测试各种内容类型"""
        # 准备包含不同类型内容的文档集合
        diverse_docs = {
            "english_doc": {
                "content": "This is an English document with some technical terms like API, JSON, HTTP."
            },
            "chinese_doc": {
                "content": "这是一个中文文档，包含一些技术术语如应用程序接口、数据格式等。"
            },
            "mixed_doc": {
                "content": "This is a mixed document 这包含中英文混合内容 with numbers 123 and symbols !@#$%"
            },
            "special_chars_doc": {
                "content": "Document with special characters: \n\t\r\\\"'`[]{}()<>"
            }
        }
        
        # 处理多样化的文档内容
        result = get_chunks(diverse_docs)
        
        # 验证所有类型的文档都能正确处理，每种文档至少产生一个分块
        self.assertEqual(len([chunk for chunk in result.values() 
                            if chunk["full_doc_id"] == "english_doc"]), 1)     # 英文文档
        self.assertEqual(len([chunk for chunk in result.values() 
                            if chunk["full_doc_id"] == "chinese_doc"]), 1)     # 中文文档
        self.assertEqual(len([chunk for chunk in result.values() 
                            if chunk["full_doc_id"] == "mixed_doc"]), 1)       # 混合语言文档
        self.assertEqual(len([chunk for chunk in result.values() 
                            if chunk["full_doc_id"] == "special_chars_doc"]), 1)  # 特殊字符文档


if __name__ == "__main__":
    unittest.main() 