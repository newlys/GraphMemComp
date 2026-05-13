#!/usr/bin/env python
"""Delete events without edges."""

import requests
import json

url = "http://127.0.0.1:8000/api/graph"

print("获取图谱数据...")
graph = requests.get(url, timeout=10).json()

subgraphs = graph.get('subgraphs', {})
events_without_edges = []

for event_id, sub in subgraphs.items():
    total_edges = (
        len(sub.get('entity_edges', [])) +
        len(sub.get('temporal_edges', [])) +
        len(sub.get('causal_edges', []))
    )
    if total_edges == 0:
        events_without_edges.append(event_id)

print(f"\n找到 {len(events_without_edges)} 个没有边的事件")

for event_id in events_without_edges[:10]:  # Delete first 10
    print(f"删除 {event_id}...")
    try:
        r = requests.delete(f"http://127.0.0.1:8000/api/nodes/{event_id}", timeout=10)
        if r.status_code == 200:
            print(f"  [OK]")
        else:
            print(f"  [ERROR] {r.status_code}")
    except Exception as e:
        print(f"  [ERROR] {e}")

print(f"\n剩余 {len(events_without_edges) - 10} 个未删除（可手动删除）")

# Final check
print("\n=== 最终状态 ===")
graph = requests.get(url, timeout=10).json()
print(f"事件数: {graph.get('coarse_event_count')}")

subgraphs = graph.get('subgraphs', {})
ok_count = 0
for event_id, sub in subgraphs.items():
    total = len(sub.get('entity_edges', [])) + len(sub.get('temporal_edges', [])) + len(sub.get('causal_edges', []))
    if total > 0:
        ok_count += 1
        
print(f"有边的事件: {ok_count}")
