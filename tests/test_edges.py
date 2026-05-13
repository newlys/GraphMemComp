#!/usr/bin/env python
"""Test edge creation in memory."""

import sys
import json

# Import the memory module
from api import get_memory, build_memory

print("初始化内存...")
memory = get_memory()

print(f"\n当前状态:")
print(f"  图已初始化: {memory.graph_initialized}")
print(f"  事件数: {len(memory.coarse_order)}")
print(f"  实体节点: {len(memory.entity_nodes)}")
print(f"  实体边: {len(memory.entity_edges)}")

if memory.entity_edges:
    print(f"\n边数据示例:")
    for edge_id, edge in list(memory.entity_edges.items())[:3]:
        print(f"  {edge_id[:40]}: {edge.source_id[:8]} -> {edge.target_id[:8]}")
else:
    print(f"\n[WARNING] 没有边数据!")
    print(f"  entity_edges 字典大小: {len(memory.entity_edges)}")
    print(f"  entity_nodes 字典大小: {len(memory.entity_nodes)}")
    
    # Check if there are any events
    if memory.coarse_order:
        event_id = memory.coarse_order[0]
        event = memory.coarse_events[event_id]
        print(f"\n第一个事件: {event_id}")
        print(f"  实体节点IDs: {event.entity_node_ids[:5]}")
        
        # Check if those nodes exist
        for node_id in event.entity_node_ids[:3]:
            if node_id in memory.entity_nodes:
                print(f"    节点 {node_id[:8]} 存在")
            else:
                print(f"    节点 {node_id[:8]} 不存在!")
