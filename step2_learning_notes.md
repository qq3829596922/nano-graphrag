# GraphRAG学习系列 - 第二步：文本处理系统

## 📖 学习目标

这一步我们深入理解GraphRAG中的**文本分块（Text Chunking）**机制，这是构建知识图谱的关键第二步。

### 🎯 核心概念

1. **为什么需要分块？**
   - LLM有上下文长度限制（如GPT-4的8K-32K tokens）
   - 长文档需要切分成适合处理的小段落
   - 分块质量直接影响后续实体提取和关系识别的效果

2. **分块的挑战**
   - 如何保持语义完整性？
   - 如何防止关键信息在边界丢失？
   - 如何平衡分块大小和处理效率？

## 🔧 技术实现

### 核心类结构

```
TextProcessor (主处理器)
├── SeparatorSplitter (分隔符分割器)
├── ChunkResult (分块结果数据结构)
└── 两种分块策略:
    ├── chunking_by_token_size() (Token分块)
    └── chunking_by_separators() (分隔符分块)
```

### 关键技术特性

1. **精确Token计算**
   - 使用tiktoken库精确计算token数量
   - 支持多种LLM模型的编码格式
   - 批量编码优化性能

2. **智能重叠机制**
   - 滑动窗口避免信息丢失
   - 可配置重叠大小
   - 保持上下文连贯性

3. **多层分隔符策略**
   ```python
   优先级分隔符:
   "\n\n"    # 段落分隔（最高优先级）
   "\n"      # 行分隔
   "。"/"."  # 句号
   "！"/"!"  # 感叹号
   "？"/"?"  # 问号
   " "       # 空格（最低优先级）
   ```

## 🏃 运行步骤

### 1. 安装依赖
```bash
pip install tiktoken
```

### 2. 运行演示
```bash
python step2_text_processing.py
```

### 3. 观察输出结果
- Token分块 vs 分隔符分块的对比
- 分块大小和重叠的效果
- 与存储系统的集成

## 📊 学习重点

### 1. Token分块策略
```python
# 滑动窗口机制
step_size = max_token_size - overlap_token_size
for start in range(0, len(tokens), step_size):
    chunk = tokens[start:start + max_token_size]
```

**特点：**
- ✅ 精确控制大小
- ✅ 处理速度快
- ❌ 可能截断语义

### 2. 分隔符分块策略
```python
# 智能语义边界识别
splitter = SeparatorSplitter(
    separators=separator_tokens,
    chunk_size=max_token_size,
    chunk_overlap=overlap_token_size,
)
```

**特点：**
- ✅ 保持语义完整
- ✅ 智能边界对齐
- ❌ 处理稍慢

### 3. 数据结构设计
```python
@dataclass
class ChunkResult:
    content: str          # 分块内容
    token_count: int      # Token数量
    chunk_index: int      # 在原文档中的顺序
    doc_id: str          # 原始文档ID
    chunk_id: str        # 分块唯一ID
```

## 🔍 代码分析重点

### 1. 批量处理优化
```python
# 批量编码提高性能
tokens_list = self.encoder.encode_batch(contents, num_threads=16)

# 批量解码
chunk_contents = self.encoder.decode_batch(chunk_tokens_list)
```

### 2. 重叠机制实现
```python
def _add_overlap(self, chunks):
    result = [chunks[0]]  # 第一个分块不需要重叠
    for i in range(1, len(chunks)):
        overlap_tokens = chunks[i-1][-self._chunk_overlap:]
        new_chunk = overlap_tokens + chunks[i]
        result.append(new_chunk)
```

### 3. 与存储系统集成
```python
async def save_chunks_to_storage(self, chunks, storage_dir):
    from step1_document_processing import JsonKVStorage
    chunk_storage = JsonKVStorage(namespace="text_chunks", ...)
    # 保存到存储系统
```

## 🧪 实验建议

### 1. 参数调优实验
```python
# 实验不同的分块大小
processor = TextProcessor(
    default_chunk_size=800,   # 试试 400, 800, 1200
    default_overlap_size=80   # 试试 50, 100, 150
)
```

### 2. 分隔符定制实验
```python
# 尝试不同的分隔符优先级
custom_separators = [
    "\n\n",     # 段落优先
    "。",       # 中文句号
    ".",        # 英文句号
    " ",        # 空格
]
```

### 3. 性能对比实验
- 测试不同文档长度的处理时间
- 比较两种分块策略的效果
- 观察重叠大小对结果质量的影响

## 🔗 与原始项目的对应关系

| 我们的实现 | 原始项目位置 | 功能对应 |
|-----------|-------------|----------|
| `TextProcessor` | `nano_graphrag/_op.py` | 文本处理核心逻辑 |
| `chunking_by_token_size()` | `chunking_by_token_size()` | Token分块方法 |
| `SeparatorSplitter` | `nano_graphrag/_splitter.py` | 分隔符分割器 |
| `ChunkResult` | 分块数据结构 | 结果格式定义 |

## ❓ 思考题

1. **为什么需要重叠机制？**
   - 防止重要信息在分块边界丢失
   - 保持上下文连贯性
   - 提高后续处理的准确性

2. **如何选择合适的分块大小？**
   - 考虑LLM的上下文限制
   - 平衡处理效率和效果质量
   - 根据文档类型调整策略

3. **两种分块策略各适用什么场景？**
   - Token分块：严格大小控制，批量处理
   - 分隔符分块：语义完整性要求高的场景

## ✅ 检查点

完成这一步后，你应该能够：
- [ ] 理解两种分块策略的原理和差异
- [ ] 掌握tiktoken的使用方法
- [ ] 了解重叠机制的作用
- [ ] 能够根据需求选择合适的分块参数
- [ ] 理解分块质量对后续处理的影响

## 🚀 下一步预告

**第三步：实体提取系统**
- 使用LLM从文本分块中提取实体和关系
- 实现提示词工程和输出解析
- 构建实体-关系数据结构
- 为知识图谱构建做准备 