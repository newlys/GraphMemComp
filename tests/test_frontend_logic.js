// 测试前端逻辑
const data = {
  "events": [
    {
      "id": "event_460fd1",
      "title": "东区机房起火，烟雾扩散蔓延",
      "entity_count": 71,
      "temporal_count": 8,
      "causal_count": 5
    }
  ],
  "subgraphs": {
    "event_460fd1": {
      "entity_nodes": [{"id": "entity_a43c1", "name": "消防员", "entity_type": "person"}],
      "temporal_nodes": [],
      "causal_nodes": []
    }
  }
};

const nodeData = data.events[0];
const subgraphs = data.subgraphs;

console.log("节点数据:", nodeData);
console.log("子图数据:", subgraphs);

const hasSubgraph = subgraphs?.[nodeData.id] && 
  ((subgraphs[nodeData.id].entity_nodes?.length || 0) > 0 || 
   (subgraphs[nodeData.id].temporal_nodes?.length || 0) > 0 || 
   (subgraphs[nodeData.id].causal_nodes?.length || 0) > 0);

console.log("hasSubgraph:", hasSubgraph);
console.log("应该显示按钮:", hasSubgraph ? "YES" : "NO");
