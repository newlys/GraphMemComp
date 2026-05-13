#!/usr/bin/env python
"""Clear database and re-seed with data that ensures all events have edges."""

import shutil
import requests
import json
import time

# Clear old data
print("正在清空数据库...")
shutil.rmtree('qdrant_data', ignore_errors=True)
print("数据库已清空\n")

# Restart server would be needed, but we'll just seed
# The server should auto-initialize on next request

url = "http://127.0.0.1:8000/api/seed"

# Use fewer questions to create fewer, cleaner events
# Each question should have 3-8 entities to trigger edge creation
payload = {
    "questions": [
        "东区三楼机房出现明火，热成像显示北侧墙面温度78度，有化学品存储柜，消防员正在疏散",
        "指挥员要求关闭东区电源，部署两支水枪从南侧楼梯推进，启动排烟系统",
        "二楼仓库发现易燃物品泄漏，需要紧急疏散周围人员，消防队已到达",
        "消防队从东侧入口建立水源，连接消防栓，准备灭火",
        "烟雾向疏散通道蔓延，2名人员可能受困在302室，已通知救援队"
    ]
}

print("正在注入新的种子数据...")
try:
    response = requests.post(url, json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print("\n[OK] 种子数据插入成功！")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Check graph data
        print("\n正在验证图谱数据...")
        graph_url = "http://127.0.0.1:8000/api/graph"
        graph_response = requests.get(graph_url, timeout=30)
        
        if graph_response.status_code == 200:
            snapshot = graph_response.json()
            print(f"\n图谱状态:")
            print(f"  - 已初始化: {snapshot.get('initialized')}")
            print(f"  - 事件数: {snapshot.get('coarse_event_count')}")
            print(f"  - 实体节点: {snapshot.get('entity_node_count')}")
            print(f"  - 时序节点: {snapshot.get('temporal_node_count')}")
            print(f"  - 因果节点: {snapshot.get('causal_node_count')}")
            
            subgraphs = snapshot.get('subgraphs', {})
            print(f"\n各事件边数统计:")
            for event_id, subgraph in subgraphs.items():
                entity_edges = len(subgraph.get('entity_edges', []))
                temporal_edges = len(subgraph.get('temporal_edges', []))
                causal_edges = len(subgraph.get('causal_edges', []))
                total = entity_edges + temporal_edges + causal_edges
                
                status = "[OK]" if total > 0 else "[WARNING]"
                print(f"  {status} 事件 {event_id[:12]}: 实体边={entity_edges}, 时序边={temporal_edges}, 因果边={causal_edges}")
        else:
            print(f"[ERROR] 获取图谱失败: {graph_response.status_code}")
    else:
        print(f"[ERROR] 请求失败: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("[ERROR] 无法连接到服务器，请先启动服务器: python main.py")
except Exception as e:
    print(f"[ERROR] {e}")
