import os  # 导入操作系统路径相关模块
from dataclasses import dataclass  # 导入数据类装饰器

from .._utils import load_json, logger, write_json  # 导入JSON读写工具和日志记录器
from ..base import (  # 导入基础存储类
    BaseKVStorage,  # 键值存储基类
)


@dataclass  # 使用dataclass装饰器定义数据类
class JsonKVStorage(BaseKVStorage):  # JSON键值存储类，继承自基础键值存储类
    def __post_init__(self):  # 初始化后处理方法，在dataclass初始化完成后自动调用
        working_dir = self.global_config["working_dir"]  # 从全局配置中获取工作目录路径
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")  # 构建JSON文件的完整路径
        self._data = load_json(self._file_name) or {}  # 从文件加载JSON数据，如果文件不存在则初始化为空字典
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")  # 记录加载的键值对数量日志

    async def all_keys(self) -> list[str]:  # 异步方法：获取所有键的列表
        return list(self._data.keys())  # 返回内存中数据的所有键组成的列表

    async def index_done_callback(self):  # 异步方法：索引完成后的回调函数
        write_json(self._data, self._file_name)  # 将内存中的数据写入到JSON文件中

    async def get_by_id(self, id):  # 异步方法：根据ID获取单个数据项
        return self._data.get(id, None)  # 从内存数据中获取指定ID的值，不存在则返回None

    async def get_by_ids(self, ids, fields=None):  # 异步方法：根据ID列表批量获取数据，可选择特定字段
        if fields is None:  # 如果没有指定字段
            return [self._data.get(id, None) for id in ids]  # 返回完整的数据项列表
        return [  # 如果指定了字段，则只返回指定字段的数据
            (
                {k: v for k, v in self._data[id].items() if k in fields}  # 筛选出指定字段的键值对
                if self._data.get(id, None)  # 如果数据项存在
                else None  # 如果数据项不存在则返回None
            )
            for id in ids  # 遍历所有ID
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:  # 异步方法：过滤出不存在于存储中的键
        return set([s for s in data if s not in self._data])  # 返回输入列表中不在当前存储中的键的集合

    async def upsert(self, data: dict[str, dict]):  # 异步方法：插入或更新数据
        self._data.update(data)  # 将新数据更新到内存中的数据字典

    async def drop(self):  # 异步方法：清空存储中的所有数据
        self._data = {}  # 将内存数据重置为空字典
