#!/usr/bin/env python
"""Check if data was seeded successfully."""

import requests
import json
import time

print("等待服务器处理...")
time.sleep(5)

url = "http://127.0.0.1:8000/api/graph"

try:
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n=== 图谱状态 ===")
        print(f"已初始化: {data.get('initialized')}")
        print(f"事件数: {data.get('coarse_event_count')}")
        print(f"实体节点: {data.get('entity_node_count')}")
        print(f"时序节点: {data.get('temporal_node_count')}")
        print(f"因果节点: {data.get('causal_node_count')}")
        
        subgraphs = data.get('subgraphs', {})
        print(f"\n=== 各事件边数 ===")
        
        all_have_edges = True
        for event_id, subgraph in subgraphs.items():
            entity_edges = len(subgraph.get('entity_edges', []))
            temporal_edges = len(subgraph.get('temporal_edges', []))
            causal_edges = len(subgraph.get('causal_edges', []))
            total = entity_edges + temporal_edges + causal_edges
            
            status = "[OK]" if total > 0 else "[NO EDGES]"
            if total == 0:
                all_have_edges = False
            
            print(f"  {status} {event_id[:12]}: 实体边={entity_edges}, 时序边={temporal_edges}, 因果边={causal_edges}")
            
            # 打印示例边
            if subgraph.get('entity_edges'):
                edge = subgraph['entity_edges'][0]
                print(f"         示例: {edge.get('source_id', '?')[:8]} -> {edge.get('target_id', '?')[:8]}")
        
        if all_have_edges:
            print("\n[SUCCESS] 所有事件都有边！")
        else:
            print("\n[WARNING] 部分事件没有边")
    else:
        print(f"[ERROR] 请求失败: {response.status_code}")
        
except Exception as e:
    print(f"[ERROR] {e}")
