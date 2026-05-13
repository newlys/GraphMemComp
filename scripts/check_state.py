#!/usr/bin/env python
"""Check current graph state and event distribution."""

import requests
import json

url = "http://127.0.0.1:8000/api/graph"

print("获取图谱数据...")
graph = requests.get(url, timeout=10).json()

print(f"\n=== 总体统计 ===")
print(f"事件数: {graph.get('coarse_event_count')}")
print(f"实体节点: {graph.get('entity_node_count')}")
print(f"时序节点: {graph.get('temporal_node_count')}")
print(f"因果节点: {graph.get('causal_node_count')}")
print(f"轮次: {graph.get('total_turn_count')}")

events = graph.get('events', [])
print(f"\n=== 事件详情 ({len(events)} 个) ===")
for event in events:
    eid = event['id'][:12]
    title = event['title'][:30]
    turns = event['turn_count']
    entities = event['entity_count']
    temporal = event['temporal_count']
    causal = event['causal_count']
    print(f"  {eid}: {title}")
    print(f"    轮次={turns}, 实体={entities}, 时序={temporal}, 因果={causal}")
