# 项目结构

```
GraphMemComp/
├── 📁 核心代码/
│   ├── main.py              # 应用入口，启动 FastAPI 服务
│   ├── api.py               # API 路由定义
│   ├── frontend.py          # 前端可视化界面
│   ├── cli.py               # 命令行交互接口
│   ├── memory_graph_v2.py   # 双粒度多重图谱核心实现
│   ├── memory_graph.py      # 旧版图谱实现（保留）
│   └── visualizer.py        # 图谱可视化工具
│
├── 📁 配置/
│   ├── model_config_work.py # 模型配置
│   └── requirements.txt     # Python 依赖
│
├── 📁 测试脚本/
│   ├── test_api.py          # API 接口测试
│   ├── test_density.py      # 信息密度测试
│   ├── test_edges.py        # 边创建测试
│   └── test_frontend_logic.js # 前端逻辑测试
│
├── 📁 数据脚本/
│   ├── quick_seed.py        # 快速数据注入
│   ├── final_seed.py        # 最终数据注入
│   ├── reseed.py            # 重新注入数据
│   ├── seed_test.py         # 测试数据注入
│   ├── seed_rich_temporal.py # 时序丰富数据注入
│   └── wait_and_seed.py     # 延迟数据注入
│
├── 📁 调试工具/
│   ├── check_data.py        # 数据检查
│   ├── check_edges.py       # 边数据检查
│   ├── check_graph.py       # 图谱状态检查
│   ├── check_state.py       # 状态检查
│   ├── debug_retrieval.py   # 检索调试
│   ├── view_qdrant_data.py  # Qdrant 数据查看
│   └── clean_old_events.py  # 清理旧事件
│
├── 📁 文档/
│   ├── docs/                # 专利文档和流程图
│   └── generate_docs.py     # 文档生成脚本
│
├── 📁 数据存储/
│   ├── qdrant_data/         # Qdrant 向量数据库
│   └── graph_data_debug.json # 调试数据
│
└── 📁 缓存/
    └── __pycache__/         # Python 字节码缓存
```

## 核心模块说明

### 双粒度多重图谱架构
- **粗粒度层**: 事件节点，表示宏观火灾事件
- **细粒度层**: 三类子图
  - 实体-关系图：人物、地点、设备等实体及其关系
  - 时序拓扑图：时间阶段和顺序关系
  - 因果关系图：原因和结果的因果链

### 主要功能
1. 多模态交互（文本+图像）
2. 自动信息抽取（LLM驱动）
3. 动态图谱构建与更新
4. 智能检索与上下文推荐
5. 可视化展示

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py

# 访问前端
# http://127.0.0.1:8000
```
