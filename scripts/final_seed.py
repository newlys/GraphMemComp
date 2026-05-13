#!/usr/bin/env python
"""Seed data and verify."""

import requests
import time
import json

url = "http://127.0.0.1:8000"

# Wait for server
print("等待服务器...")
for i in range(20):
    try:
        r = requests.get(f"{url}/api/graph", timeout=5)
        if r.status_code == 200:
            print("服务器就绪！")
            break
    except:
        time.sleep(2)

# Seed data
print("\n注入数据...")
seed_data = {
    "questions": [
        "东区三楼机房起火，热成像显示北侧墙面78度，有化学品柜，消防员疏散人员",
        "指挥员要求关闭东区电源，部署两支水枪从南侧楼梯推进灭火，启动排烟系统",
        "二楼仓库发现易燃物品泄漏，需要紧急疏散周围人员，消防队已到达现场",
        "消防队从东侧入口建立水源，连接消防栓，准备展开灭火行动",
        "烟雾向疏散通道蔓延，2名人员可能受困在302室，已通知救援队前往"
    ]
}

try:
    r = requests.post(f"{url}/api/seed", json=seed_data, timeout=120)
    if r.status_code == 200:
        print(f"[OK] {r.json()}")
    else:
        print(f"[ERROR] {r.status_code}: {r.text}")
except Exception as e:
    print(f"[超时，但可能正在处理] {e}")

# Wait for processing
print("\n等待处理...")
time.sleep(5)

# Check
print("\n=== 图谱状态 ===")
graph = requests.get(f"{url}/api/graph", timeout=10).json()
print(f"事件数: {graph.get('coarse_event_count')}")
print(f"实体节点: {graph.get('entity_node_count')}")
print(f"时序节点: {graph.get('temporal_node_count')}")
print(f"因果节点: {graph.get('causal_node_count')}")

subgraphs = graph.get('subgraphs', {})
print(f"\n各事件详情:")
for event_id, sub in subgraphs.items():
    e_nodes = len(sub.get('entity_nodes', []))
    e_edges = len(sub.get('entity_edges', []))
    t_nodes = len(sub.get('temporal_nodes', []))
    c_nodes = len(sub.get('causal_nodes', []))
    print(f"  {event_id[:12]}: 实体={e_nodes}节点/{e_edges}边, 时序={t_nodes}节点, 因果={c_nodes}节点")
