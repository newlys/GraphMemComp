#!/usr/bin/env python
"""Test API and verify subgraph data."""

import requests
import json

url = "http://127.0.0.1:8000/api/graph"

print("正在获取图谱数据...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    
    print(f"\n=== 图谱基本信息 ===")
    print(f"已初始化: {data.get('initialized')}")
    print(f"事件数: {data.get('coarse_event_count')}")
    print(f"实体节点: {data.get('entity_node_count')}")
    print(f"时序节点: {data.get('temporal_node_count')}")
    print(f"因果节点: {data.get('causal_node_count')}")
    
    print(f"\n=== 事件列表 ===")
    for event in data.get('events', []):
        print(f"  事件ID: {event['id'][:12]}...")
        print(f"    标题: {event['title']}")
        print(f"    实体数: {event['entity_count']}")
        print(f"    时序数: {event['temporal_count']}")
        print(f"    因果数: {event['causal_count']}")
    
    print(f"\n=== 子图数据 ===")
    subgraphs = data.get('subgraphs', {})
    for event_id, subgraph in subgraphs.items():
        print(f"  事件 {event_id[:12]}...:")
        print(f"    实体节点数: {len(subgraph.get('entity_nodes', []))}")
        print(f"    时序节点数: {len(subgraph.get('temporal_nodes', []))}")
        print(f"    因果节点数: {len(subgraph.get('causal_nodes', []))}")
        
        # 打印前2个实体节点
        if subgraph.get('entity_nodes'):
            print(f"    示例实体节点:")
            for node in subgraph['entity_nodes'][:2]:
                print(f"      - ID: {node['id'][:12]}..., 名称: {node['name']}, 类型: {node['entity_type']}")
    
    # 保存完整数据供前端调试
    with open('graph_data_debug.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n完整数据已保存到 graph_data_debug.json")
    
else:
    print(f"[ERROR] 请求失败: {response.status_code}")
    print(response.text)
