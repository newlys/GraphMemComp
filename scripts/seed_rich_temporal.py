#!/usr/bin/env python
"""Seed data with rich temporal and causal information."""

import requests
import json
import time

url = "http://127.0.0.1:8000/api/seed"

# Create a coherent fire incident with clear temporal progression and causal chains
payload = {
    "questions": [
        # Phase 1: Initial discovery (10:00)
        "10点整，东区三楼机房烟感报警器首次触发，热成像显示北侧墙面温度65度，有少量烟雾",
        
        # Phase 2: Fire development (10:05)
        "10点05分，火势蔓延至化学品存储柜，温度升至78度，浓烟向走廊扩散，3名人员开始疏散",
        
        # Phase 3: Emergency response (10:10)
        "10点10分，指挥员到达现场，要求关闭东区总电源，部署第一支水枪从南侧楼梯推进",
        
        # Phase 4: Rescue operations (10:15)
        "10点15分，发现2名人员受困302室，启动应急广播，第二支水枪掩护救援队进入",
        
        # Phase 5: Fire suppression (10:25)
        "10点25分，消防队建立东侧水源，连接3个消防栓，启动排烟系统，温度降至60度",
        
        # Phase 6: Situation control (10:40)
        "10点40分，明火基本扑灭，温度降至45度，完成人员清点，开始现场清理",
        
        # Additional causal information
        "因化学品泄漏导致复燃风险，持续监控北侧墙面温度，保持水枪待命",
        
        "由于排烟系统启动成功，走廊能见度恢复，救援队顺利进入302室救出受困人员",
    ]
}

print("注入富含时序和因果的数据...")
try:
    r = requests.post(url, json=payload, timeout=180)
    if r.status_code == 200:
        result = r.json()
        print(f"[OK] {json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        print(f"[ERROR] {r.status_code}")
        print(r.text[:200])
except Exception as e:
    print(f"[可能超时但正在处理] {e}")

print("\n等待处理...")
time.sleep(5)

print("\n检查图谱...")
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
    
    # 打印示例时序节点
    if sub.get('temporal_nodes'):
        print(f"    时序节点示例:")
        for node in sub['temporal_nodes'][:3]:
            print(f"      - {node.get('stage_name', '?')}: {node.get('summary', '?')[:30]}")
