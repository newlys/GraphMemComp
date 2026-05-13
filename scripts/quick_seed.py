#!/usr/bin/env python
"""Seed data via API."""

import requests
import json
import time

url = "http://127.0.0.1:8000/api/seed"

payload = {
    "questions": [
        "东区三楼机房起火，热成像显示北侧墙面78度，有化学品柜，消防员疏散人员",
        "指挥员关闭东区电源，部署两支水枪从南侧楼梯推进灭火，启动排烟系统",
        "二楼仓库易燃品泄漏，紧急疏散人员，消防队到达现场",
        "消防队从东侧入口建立水源，连接消防栓准备灭火",
        "烟雾向疏散通道蔓延，2人受困302室，通知救援队"
    ]
}

print("注入数据...")
try:
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code == 200:
        print(f"[OK] {r.json()}")
    else:
        print(f"[ERROR] {r.status_code}")
except Exception as e:
    print(f"[可能超时但正在处理] {e}")

time.sleep(5)

print("\n检查图谱...")
g = requests.get("http://127.0.0.1:8000/api/graph", timeout=10).json()
print(f"事件: {g.get('coarse_event_count')}")
print(f"实体: {g.get('entity_node_count')}")

subs = g.get('subgraphs', {})
for eid, sub in subs.items():
    edges = len(sub.get('entity_edges', []))
    nodes = len(sub.get('entity_nodes', []))
    print(f"  {eid[:12]}: {nodes}节点, {edges}边")
