#!/usr/bin/env python
"""Seed test data to trigger cold-start and populate fine-grained graphs."""

import requests
import json

url = "http://127.0.0.1:8000/api/seed"

payload = {
    "questions": [
        "东区三楼机房出现明火，热成像显示北侧墙面温度达到78摄氏度，有化学品存储柜",
        "三楼北侧烟雾向疏散通道蔓延，2名人员可能受困在302室，已通知消防队",
        "指挥员要求关闭东区电源，并部署两支水枪从南侧楼梯推进，启动排烟系统",
        "二楼仓库发现易燃物品泄漏，需要紧急疏散周围人员",
        "消防队到达现场，开始从东侧入口建立水源"
    ]
}

print("正在发送种子数据...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print("\n[OK] 种子数据插入成功！")
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"\n[ERROR] 请求失败: {response.status_code}")
    print(response.text)

# Now check the graph snapshot
print("\n\n正在获取图谱快照...")
graph_url = "http://127.0.0.1:8000/api/graph"
graph_response = requests.get(graph_url)

if graph_response.status_code == 200:
    snapshot = graph_response.json()
    print(f"\n图谱状态:")
    print(f"  - 已初始化: {snapshot.get('initialized')}")
    print(f"  - 粗粒度事件数: {snapshot.get('coarse_event_count')}")
    print(f"  - 实体节点数: {snapshot.get('entity_node_count')}")
    print(f"  - 时序节点数: {snapshot.get('temporal_node_count')}")
    print(f"  - 因果节点数: {snapshot.get('causal_node_count')}")
    print(f"  - 压缩率: {snapshot.get('compression_ratio')}")
    
    if snapshot.get('subgraphs'):
        print(f"\n子图数据:")
        for event_id, subgraph in snapshot['subgraphs'].items():
            print(f"  事件 {event_id[:8]}...:")
            print(f"    - 实体节点: {len(subgraph.get('entity_nodes', []))}")
            print(f"    - 时序节点: {len(subgraph.get('temporal_nodes', []))}")
            print(f"    - 因果节点: {len(subgraph.get('causal_nodes', []))}")
else:
    print(f"\n[ERROR] 获取图谱失败: {graph_response.status_code}")
