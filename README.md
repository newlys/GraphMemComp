# GraphMemComp 🧠🔥

> **基于双粒度多重图谱的火场记忆压缩系统**  
> 多模态交互 · LLM驱动抽取 · 动态图谱压缩 · 智能检索

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Qdrant](https://img.shields.io/badge/Vector-Qdrant-4A90E2?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iIzRBOTBFMiIgZD0iTTEyIDJMMiA3bDEwIDUgMTAtNUwtMi0yeiIvPjwvc3ZnPg==)](https://qdrant.tech/)

---

## ✨ 特性亮点

- 🎯 **双粒度架构** - 粗粒度事件层 + 细粒度子图(实体关系/时序拓扑/因果关系)
- 🤖 **LLM 驱动** - 基于通义千问的自动信息抽取,无需人工标注
- 🔍 **多维检索** - 语义相似度 + 实体匹配 + 时间衰减 + 关键词加权
- 📊 **动态压缩** - LLM在线融合 + 主动权重衰减,实现记忆压缩
- 🌐 **可视化前端** - 交互式图谱展示,支持子图切换和节点详情
- 💾 **持久化存储** - 基于 Qdrant 向量数据库,支持高效相似度搜索
- 📱 **多模态支持** - 文本描述 + 图像证据联合处理

---

## 🚀 快速开始

### 1️⃣ 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/GraphMemComp.git
cd GraphMemComp

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 配置 API Key

```bash
# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="your-api-key-here"

# 或创建 .env 文件 (需要 python-dotenv)
echo "DASHSCOPE_API_KEY=your-api-key-here" > .env
```


### 3️⃣ 启动服务

```bash
# 启动 Web 服务
python main.py

# 可选: 注入示例数据
python main.py --seed

# 可选: 单次命令行对话
python main.py --question "东区三楼机房出现明火"
```

🌐 访问前端: **http://127.0.0.1:8000**

---

## 📐 架构设计

### 双粒度图谱结构

```
┌─────────────────────────────────────────────────────┐
│                粗粒度事件层                           │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐    │
│  │ 事件 A   │────▶│ 事件 B   │────▶│ 事件 C   │    │
│  │ 东区起火 │     │ 电源关闭 │     │ 灭火完成 │    │
│  └────┬─────┘     └──────────┘     └──────────┘    │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────────────────┐   │
│  │        细粒度子图 (事件A)                     │   │
│  │  ┌────────────┐  ┌────────────┐             │   │
│  │  │ 实体关系图  │  │ 时序拓扑图  │             │   │
│  │  │ • 人物      │  │ • 时间窗口  │             │   │
│  │  │ • 地点      │  │ • 阶段过渡  │             │   │
│  │  │ • 设备      │  │ • 传感器状态│             │   │
│  │  │ • 危险品    │  │             │             │   │
│  │  └────────────┘  └────────────┘             │   │
│  │  ┌────────────┐                              │   │
│  │  │ 因果关系图  │                              │   │
│  │  │ • 触发条件  │                              │   │
│  │  │ • 决策动作  │                              │   │
│  │  │ • 结果状态  │                              │   │
│  │  └────────────┘                              │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 数据处理流程

```
用户输入 ──▶ LLM抽取 ──▶ 图谱构建 ──▶ 向量存储 ──▶ 多维检索
  │            │            │            │            │
  ▼            ▼            ▼            ▼            ▼
文本/图像   实体/时序/   节点创建    Qdrant     语义+实体
          因果关系    边建立      索引      +时间检索
```

---

## 📡 API 文档

### 核心接口

| 端点 | 方法 | 描述 | 示例 |
|------|------|------|------|
| `/api/chat` | `POST` | 发送交互,写入图谱 | 发送火场报告 |
| `/api/graph` | `GET` | 获取图谱快照 | 查看完整图谱 |
| `/api/seed` | `POST` | 注入示例数据 | 初始化测试数据 |
| `/api/nodes/{id}` | `DELETE` | 删除节点 | 清理错误节点 |
| `/api/turns/{id}` | `DELETE` | 删除对话轮次 | 删除历史交互 |
| `/api/debug` | `GET` | 调试信息 | 查看系统状态 |

### 请求示例

#### 发送火场报告

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "东区三楼机房出现明火,热成像显示北侧墙面温度达到78摄氏度",
    "image_paths": ["/path/to/thermal_image.jpg"],
    "visual_description": "热成像图显示高温区域",
    "saliency_score": 0.85
  }'
```

#### 获取图谱数据

```bash
curl http://127.0.0.1:8000/api/graph | jq '.coarse_event_count'
```

---

## 📁 项目结构

```
GraphMemComp/
├── 📄 核心模块
│   ├── main.py                 # 应用入口 (FastAPI/CLI)
│   ├── api.py                  # REST API 路由
│   ├── frontend.py             # 前端可视化界面
│   ├── cli.py                  # 命令行交互接口
│   ├── memory_graph_v2.py      # 双粒度图谱核心实现 (3816行)
│   ├── memory_graph.py         # 旧版图谱实现 (向后兼容)
│   └── visualizer.py           # 图谱可视化工具
│
├── ⚙️ 配置与依赖
│   ├── model_config_work.py    # 多模态模型配置
│   ├── requirements.txt        # Python 依赖
│   └── .gitignore              # Git 忽略规则
│
├── 🧪 测试
│   └── tests/
│       ├── test_api.py         # API 接口测试
│       ├── test_edges.py       # 边创建测试
│       └── test_density.py     # 信息密度测试
│
├── 🛠️ 工具脚本
│   └── scripts/
│       ├── quick_seed.py       # 快速数据注入
│       ├── wait_and_seed.py    # 等待并注入数据
│       ├── check_graph.py      # 图谱状态检查
│       └── debug_retrieval.py  # 检索调试
│
└── 📊 数据目录 (Git忽略)
    ├── qdrant_data/            # Qdrant 向量数据库
    ├── graph_data_debug.json   # 调试数据
    └── memory_graph.png        # 图谱可视化
```

详细结构请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 🔧 高级用法

### 自定义模型配置

编辑 `model_config_work.py` 配置不同模型

### 检索参数调优

```python
memory = GraphMemory(
    cold_start_threshold=3,        # 冷启动阈值
    new_event_threshold=0.85,      # 新事件判定阈值
    entity_merge_threshold=0.65,   # 实体合并阈值
    temporal_merge_threshold=0.75, # 时序合并阈值
    decay_half_life_seconds=21600, # 衰减半衰期(6小时)
    compress_threshold=0.16,       # 压缩触发阈值
)
```

---

## 📊 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 压缩率 | ~0.24 | 原始交互压缩至24% |
| 空间节省 | ~76% | 向量存储优化 |
| 检索延迟 | <50ms | 相似度搜索 |
| 支持交互数 | 10,000+ | 持久化存储 |

---

## 🧪 测试

```bash
# 运行 API 测试
python tests/test_api.py

# 测试信息密度计算
python tests/test_density.py

# 验证边创建逻辑
python tests/test_edges.py
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request!

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📝 许可证

本项目采用 [MIT License](LICENSE) 开源协议

---

## 🙏 致谢

- [Qdrant](https://qdrant.tech/) - 向量数据库
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [NetworkX](https://networkx.org/) - 图计算库

---


用 ❤️ 打造 | ⭐ 如果这个项目对你有帮助,请给个 Star!

