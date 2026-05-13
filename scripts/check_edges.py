#!/usr/bin/env python
"""Check edge data in subgraphs."""

import requests
import json

url = "http://127.0.0.1:8000/api/graph"

print("正在获取图谱数据...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    
    print(f"\n=== 检查边数据 ===")
    subgraphs = data.get('subgraphs', {})
    
    for event_id, subgraph in subgraphs.items():
        print(f"\n事件 {event_id[:12]}...:")
        print(f"  实体边数: {len(subgraph.get('entity_edges', []))}")
        print(f"  时序边数: {len(subgraph.get('temporal_edges', []))}")
        print(f"  因果边数: {len(subgraph.get('causal_edges', []))}")
        
        # 打印前2条边
        if subgraph.get('entity_edges'):
            print(f"  示例实体边:")
            for edge in subgraph['entity_edges'][:2]:
                print(f"    - {edge.get('source_id', '?')[:8]} -> {edge.get('target_id', '?')[:8]}, 关系: {edge.get('relation_label', '?')}")
        
        if subgraph.get('temporal_edges'):
            print(f"  示例时序边:")
            for edge in subgraph['temporal_edges'][:2]:
                print(f"    - {edge.get('source_id', '?')[:8]} -> {edge.get('target_id', '?')[:8]}, 关系: {edge.get('relation_label', '?')}")
        
        if subgraph.get('causal_edges'):
            print(f"  示例因果边:")
            for edge in subgraph['causal_edges'][:2]:
                print(f"    - {edge.get('source_id', '?')[:8]} -> {edge.get('target_id', '?')[:8]}, 关系: {edge.get('relation_label', '?')}")
    
else:
    print(f"[ERROR] 请求失败: {response.status_code}")
