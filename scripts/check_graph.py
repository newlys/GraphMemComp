#!/usr/bin/env python
"""Quick graph status check."""
import requests

g = requests.get("http://127.0.0.1:8000/api/graph", timeout=10).json()
print(f"事件: {g.get('coarse_event_count')}")
print(f"实体: {g.get('entity_node_count')}")
print(f"时序: {g.get('temporal_node_count')}")
print(f"因果: {g.get('causal_node_count')}")

subs = g.get('subgraphs', {})
for eid, sub in subs.items():
    e_nodes = len(sub.get('entity_nodes', []))
    e_edges = len(sub.get('entity_edges', []))
    t_nodes = len(sub.get('temporal_nodes', []))
    t_edges = len(sub.get('temporal_edges', []))
    c_nodes = len(sub.get('causal_nodes', []))
    c_edges = len(sub.get('causal_edges', []))
    print(f"\n  {eid[:12]}:")
    print(f"    实体: {e_nodes}节点, {e_edges}边")
    print(f"    时序: {t_nodes}节点, {t_edges}边")
    print(f"    因果: {c_nodes}节点, {c_edges}边")
    
    if sub.get('temporal_nodes'):
        print(f"    时序节点:")
        for node in sub['temporal_nodes'][:5]:
            print(f"      - {node.get('stage_name', '?')}: {node.get('summary', '?')[:40]}")
    
    if sub.get('causal_nodes'):
        print(f"    因果节点:")
        for node in sub['causal_nodes'][:5]:
            print(f"      - {node.get('causal_type', '?')}: {node.get('text', '?')[:40]}")
