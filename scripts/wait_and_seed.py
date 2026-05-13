#!/usr/bin/env python
"""Wait for server and seed data."""

import requests
import time
import json

url = "http://127.0.0.1:8000"

print("等待服务器就绪...")
for i in range(30):
    try:
        r = requests.get(f"{url}/api/graph", timeout=5)
        if r.status_code == 200:
            print("服务器已就绪！")
            break
    except:
        print(f"  等待中... ({i+1}s)")
        time.sleep(2)
else:
    print("服务器未启动！")
    exit(1)

# Check current state
graph = requests.get(f"{url}/api/graph", timeout=10).json()
print(f"\n当前状态:")
print(f"  事件数: {graph.get('coarse_event_count')}")
print(f"  实体节点: {graph.get('entity_node_count')}")

# Clear if there's old data
if graph.get('coarse_event_count', 0) > 0:
    print("\n清除旧数据...")
    requests.delete(f"{url}/api/reset", timeout=10)
    print("已清除")
    time.sleep(2)

# Seed new data
print("\n注入新数据...")
seed_url = f"{url}/api/seed"
payload = {
    "questions": [
        "东区三楼机房起火，热成像显示北侧墙面78度，有化学品柜，消防员疏散",
        "指挥员关闭东区电源，部署两支水枪从南侧楼梯推进，启动排烟",
        "二楼仓库易燃品泄漏，紧急疏散人员，消防队到达",
        "消防队从东侧入口建立水源，连接消防栓准备灭火",
        "烟雾向疏散通道蔓延，2人受困302室，通知救援队"
    ]
}

try:
    response = requests.post(seed_url, json=payload, timeout=120)
    if response.status_code == 200:
        result = response.json()
        print(f"[OK] 种子数据: {json.dumps(result, ensure_ascii=False)}")
    else:
        print(f"[ERROR] {response.status_code}: {response.text}")
except Exception as e:
    print(f"[ERROR] {e}")

# Wait for processing
print("\n等待处理完成...")
time.sleep(5)

# Final check
print("\n=== 最终图谱状态 ===")
graph = requests.get(f"{url}/api/graph", timeout=10).json()
print(f"事件数: {graph.get('coarse_event_count')}")
print(f"实体节点: {graph.get('entity_node_count')}")

subgraphs = graph.get('subgraphs', {})
print(f"\n各事件边数:")
for event_id, sub in subgraphs.items():
    e_edges = len(sub.get('entity_edges', []))
    t_edges = len(sub.get('temporal_edges', []))
    c_edges = len(sub.get('causal_edges', []))
    total = e_edges + t_edges + c_edges
    status = "[OK]" if total > 0 else "[NO EDGES]"
    print(f"  {status} {event_id[:12]}: 实体边={e_edges}, 时序边={t_edges}, 因果边={c_edges}")
