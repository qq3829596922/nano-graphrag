#!/usr/bin/env python3
"""
📊 GraphRAG 学习进度追踪器
帮助您管理模块化学习进度
"""

import json
from pathlib import Path
from datetime import datetime

class LearningTracker:
    """学习进度追踪器"""
    
    def __init__(self):
        self.progress_file = Path("learning_progress.json")
        self.modules = {
            "module1": {
                "name": "存储系统",
                "file": "module1_storage.py",
                "description": "学习KV存储、向量存储、图存储",
                "prerequisites": [],
                "estimated_time": "30分钟",
                "difficulty": "⭐⭐"
            },
            "module2": {
                "name": "文本处理", 
                "file": "module2_text_processing.py",
                "description": "学习文档分块、token计算",
                "prerequisites": ["module1"],
                "estimated_time": "45分钟",
                "difficulty": "⭐⭐⭐"
            },
            "module3": {
                "name": "实体提取",
                "file": "module3_entity_extraction.py", 
                "description": "学习如何从文本提取实体和关系",
                "prerequisites": ["module1", "module2"],
                "estimated_time": "60分钟",
                "difficulty": "⭐⭐⭐⭐"
            },
            "module4": {
                "name": "图构建与聚类",
                "file": "module4_graph_clustering.py",
                "description": "学习知识图谱构建和社区检测",
                "prerequisites": ["module1", "module3"],
                "estimated_time": "45分钟", 
                "difficulty": "⭐⭐⭐⭐"
            },
            "module5": {
                "name": "查询系统",
                "file": "module5_query_system.py",
                "description": "学习三种查询模式的实现",
                "prerequisites": ["module1", "module2", "module3", "module4"],
                "estimated_time": "50分钟",
                "difficulty": "⭐⭐⭐⭐⭐"
            },
            "module6": {
                "name": "LLM集成",
                "file": "module6_llm_integration.py",
                "description": "学习如何集成大语言模型",
                "prerequisites": [],
                "estimated_time": "40分钟",
                "difficulty": "⭐⭐⭐"
            }
        }
        
    def load_progress(self):
        """加载学习进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def save_progress(self, progress):
        """保存学习进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
            
    def show_overview(self):
        """显示学习概览"""
        print("🎯 GraphRAG 模块化学习路径")
        print("=" * 60)
        
        progress = self.load_progress()
        
        for module_id, module_info in self.modules.items():
            status = progress.get(module_id, {})
            completed = status.get('completed', False)
            
            # 状态图标
            status_icon = "✅" if completed else "⏳"
            
            # 前置条件检查
            prereq_met = all(
                progress.get(prereq, {}).get('completed', False) 
                for prereq in module_info['prerequisites']
            )
            
            available_icon = "🟢" if prereq_met else "🔴"
            
            print(f"{status_icon} {available_icon} Module {module_id[-1]}: {module_info['name']}")
            print(f"   📝 {module_info['description']}")
            print(f"   ⏱️  {module_info['estimated_time']} | 难度: {module_info['difficulty']}")
            
            if module_info['prerequisites']:
                prereq_names = [self.modules[p]['name'] for p in module_info['prerequisites']]
                print(f"   📚 前置: {', '.join(prereq_names)}")
                
            if completed:
                completed_date = status.get('completed_date', '未知')
                print(f"   ✅ 完成时间: {completed_date}")
                
            print()
            
    def mark_completed(self, module_id):
        """标记模块完成"""
        if module_id not in self.modules:
            print(f"❌ 模块 {module_id} 不存在")
            return
            
        progress = self.load_progress()
        progress[module_id] = {
            'completed': True,
            'completed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_progress(progress)
        print(f"✅ 已标记 {self.modules[module_id]['name']} 为完成")
        
    def get_next_module(self):
        """获取下一个可学习的模块"""
        progress = self.load_progress()
        
        for module_id, module_info in self.modules.items():
            # 跳过已完成的模块
            if progress.get(module_id, {}).get('completed', False):
                continue
                
            # 检查前置条件
            prereq_met = all(
                progress.get(prereq, {}).get('completed', False) 
                for prereq in module_info['prerequisites']
            )
            
            if prereq_met:
                return module_id, module_info
                
        return None, None
        
    def suggest_learning_path(self):
        """建议学习路径"""
        print("🗺️ 建议的学习路径")
        print("=" * 40)
        
        # 拓扑排序找出依赖关系
        path = []
        remaining = set(self.modules.keys())
        
        while remaining:
            # 找到没有未满足前置条件的模块
            available = []
            for module_id in remaining:
                prereqs = set(self.modules[module_id]['prerequisites'])
                if prereqs.issubset(set(path)):
                    available.append(module_id)
                    
            if not available:
                break
                
            # 按难度排序，选择最简单的
            available.sort(key=lambda x: len(self.modules[x]['difficulty']))
            next_module = available[0]
            
            path.append(next_module)
            remaining.remove(next_module)
            
        for i, module_id in enumerate(path, 1):
            module_info = self.modules[module_id]
            print(f"{i}. {module_info['name']} ({module_info['estimated_time']})")
            
        total_time = sum(int(self.modules[m]['estimated_time'].split('分')[0]) for m in path)
        print(f"\n⏱️ 总预计时间: {total_time} 分钟 ({total_time//60}小时{total_time%60}分钟)")
            
    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n📚 学习管理菜单")
            print("1. 查看学习概览")
            print("2. 获取下一个模块")
            print("3. 标记模块完成")
            print("4. 查看建议学习路径")
            print("5. 重置进度")
            print("0. 退出")
            
            choice = input("\n请选择操作 (0-5): ").strip()
            
            if choice == "1":
                self.show_overview()
            elif choice == "2":
                module_id, module_info = self.get_next_module()
                if module_id:
                    print(f"\n🎯 下一个模块: {module_info['name']}")
                    print(f"📝 描述: {module_info['description']}")
                    print(f"📄 文件: {module_info['file']}")
                    print(f"⏱️ 预计时间: {module_info['estimated_time']}")
                    
                    run_now = input("\n是否现在开始学习？(y/n): ").strip().lower()
                    if run_now == 'y':
                        print(f"💡 请运行: python {module_info['file']}")
                else:
                    print("🎉 恭喜！所有模块都已完成！")
            elif choice == "3":
                self.show_overview()
                module_id = input("\n请输入要标记完成的模块ID (如 module1): ").strip()
                self.mark_completed(module_id)
            elif choice == "4":
                self.suggest_learning_path()
            elif choice == "5":
                confirm = input("确定要重置所有进度吗？(y/n): ").strip().lower()
                if confirm == 'y':
                    if self.progress_file.exists():
                        self.progress_file.unlink()
                    print("✅ 进度已重置")
            elif choice == "0":
                print("👋 再见！祝学习愉快！")
                break
            else:
                print("❌ 无效选择，请重试")

def main():
    """主函数"""
    print("📊 GraphRAG 学习追踪器")
    print("帮助您系统化学习GraphRAG")
    
    tracker = LearningTracker()
    tracker.interactive_menu()

if __name__ == "__main__":
    main() 