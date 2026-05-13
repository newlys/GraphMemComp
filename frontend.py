"""Frontend HTML template for the fire-scene graph memory console."""

from __future__ import annotations


def build_app_html() -> str:
    """Serve a single-page UI without a separate frontend build."""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>火灾双粒度多重图谱</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {
      --page: #f6f5f0;
      --surface: #ffffff;
      --surface-alt: #f0f6f4;
      --ink: #20242a;
      --muted: #66717d;
      --line: #d9ded8;
      --accent: #b9422f;
      --accent-2: #1d6f68;
      --warn: #bf7b16;
      --blue: #33658a;
      --purple: #7a4e9f;
      --shadow: 0 8px 22px rgba(32, 36, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--page);
      min-height: 100vh;
    }
    button, textarea, input { font: inherit; }
    button {
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 9px 13px;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }
    button:hover { filter: brightness(0.96); }
    button.secondary { background: var(--blue); }
    button.ghost {
      background: var(--surface);
      color: var(--ink);
      border-color: var(--line);
    }
    button.danger { background: #b12a22; }
    textarea, input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      background: var(--surface);
      color: var(--ink);
    }
    textarea { min-height: 88px; resize: vertical; }
    input[type="range"] { width: 100%; }
    .app {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfbf8;
    }
    .brand {
      display: flex;
      align-items: baseline;
      gap: 12px;
      min-width: 260px;
    }
    .brand h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .brand span, .muted { color: var(--muted); font-size: 12px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(80px, 1fr));
      gap: 8px;
      flex: 1;
      max-width: 820px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 8px 10px;
      min-width: 0;
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
      white-space: nowrap;
    }
    .metric strong { font-size: 18px; line-height: 1.2; }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) 430px;
      gap: 14px;
      padding: 14px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
      min-width: 0;
      overflow: hidden;
    }
    .panel-head {
      min-height: 54px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .panel-title { font-size: 16px; font-weight: 700; }
    .graph-area { padding: 10px; }
    #graph {
      height: calc(100vh - 185px);
      min-height: 520px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfbf7;
    }
    .subgraph-tabs {
      display: flex;
      gap: 6px;
      padding: 8px 14px;
      border-bottom: 1px solid var(--line);
    }
    .subgraph-tab {
      padding: 5px 12px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--surface);
      cursor: pointer;
      font-size: 13px;
    }
    .subgraph-tab.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    .side {
      display: grid;
      grid-template-rows: auto minmax(190px, .8fr) minmax(210px, 1fr);
      gap: 14px;
      min-height: 0;
    }
    .section { padding: 12px 14px 14px; }
    .field { margin-bottom: 10px; }
    .field label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .row { display: flex; align-items: center; gap: 8px; }
    .row > * { min-width: 0; }
    .scroll {
      max-height: 100%;
      overflow: auto;
      padding-right: 4px;
    }
    .record {
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
    }
    .record:first-child { padding-top: 0; }
    .record:last-child { border-bottom: 0; padding-bottom: 0; }
    .record-title {
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .mono {
      font-family: Consolas, "Courier New", monospace;
      word-break: break-all;
      font-size: 12px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      border-radius: 6px;
      border: 1px solid var(--line);
      padding: 2px 6px;
      font-size: 12px;
      color: var(--muted);
      background: var(--surface-alt);
      margin: 2px 4px 2px 0;
    }
    .empty { color: var(--muted); padding: 10px 0; }
    .loading {
      display: none;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 3px;
      display: inline-block;
    }
    .legend-section {
      padding: 6px 14px;
      border-top: 1px solid var(--line);
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    @media (max-width: 1180px) {
      .topbar { align-items: stretch; flex-direction: column; }
      .brand { min-width: 0; }
      .metrics { width: 100%; max-width: none; grid-template-columns: repeat(3, 1fr); }
      .layout { grid-template-columns: 1fr; }
      #graph { height: 52vh; min-height: 380px; }
      .side { grid-template-rows: auto auto auto; }
    }
    @media (max-width: 620px) {
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .layout { padding: 8px; }
      .topbar { padding: 12px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <h1>火灾双粒度多重图谱</h1>
        <span id="viewLabel">粗粒度事件层</span>
      </div>
      <div class="metrics" id="metrics"></div>
      <div class="row">
        <button class="ghost" id="seedBtn">写入样例</button>
      </div>
    </header>

    <main class="layout">
      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title" id="graphTitle">宏观态势</div>
            <div class="muted" id="graphMeta">加载中</div>
          </div>
          <button class="secondary" id="backBtn" style="display:none;">← 返回粗粒度</button>
        </div>
        <div class="subgraph-tabs" id="subgraphTabs" style="display:none;">
          <div class="subgraph-tab active" data-tab="entity">实体-关系图</div>
          <div class="subgraph-tab" data-tab="temporal">时序拓扑图</div>
          <div class="subgraph-tab" data-tab="causal">因果关系图</div>
        </div>
        <div class="graph-area">
          <div id="graph"></div>
        </div>
        <div class="legend-section" id="legend"></div>
      </section>

      <aside class="side">
        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">交互写入</div>
            <button id="sendBtn">发送</button>
          </div>
          <div class="section">
            <div class="field">
              <label for="question">现场文本</label>
              <textarea id="question" placeholder="请输入火灾现场相关信息..."></textarea>
            </div>
            <div class="loading" id="loading">处理中...</div>
          </div>
        </section>

        <section class="panel" id="changesPanel" style="display:none;">
          <div class="panel-head">
            <div class="panel-title">本次变更</div>
            <button class="ghost" onclick="hideChanges()">关闭</button>
          </div>
          <div class="section scroll" id="changesBox"></div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">检索路径</div>
            <span class="muted" id="answerState"></span>
          </div>
          <div class="section scroll" id="retrievalBox"></div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">节点详情</div>
            <button class="danger" id="deleteNodeBtn" style="display:none;">删除事件</button>
          </div>
          <div class="section scroll" id="detailBox"></div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">历史对话</div>
            <button class="ghost" id="refreshHistoryBtn">刷新</button>
          </div>
          <div class="section scroll" id="historyBox"></div>
        </section>
      </aside>
    </main>
  </div>

  <script>
    let network = null;
    let currentGraph = { events: [], coarse_edges: [], subgraphs: {}, initialized: false };
    let currentView = "coarse";
    let currentCoarseId = null;
    let selectedNodeId = null;
    let currentSubgraphTab = "entity";

    const entityColors = {
      person: "#7a4e9f",
      location: "#33658a",
      equipment: "#1d6f68",
      material: "#8b6a4d",
      building_area: "#58778a",
      hazard: "#bf7b16"
    };
    const temporalColors = { time_window: "#33658a", stage: "#4d7c2f", sensor_state: "#697684" };
    const causalColors = { state_change: "#b9422f", trigger: "#bf7b16", decision: "#1d6f68", outcome: "#7a4e9f" };

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[ch]));
    }

    function fmt(value, digits = 2) {
      const num = Number(value || 0);
      return Number.isFinite(num) ? num.toFixed(digits) : "0.00";
    }

    function renderMetrics(snapshot) {
      const metrics = document.getElementById("metrics");
      metrics.innerHTML = `
        <div class="metric"><label>事件</label><strong>${snapshot.coarse_event_count || 0}</strong></div>
        <div class="metric"><label>实体</label><strong>${snapshot.entity_node_count || 0}</strong></div>
        <div class="metric"><label>时序</label><strong>${snapshot.temporal_node_count || 0}</strong></div>
        <div class="metric"><label>因果</label><strong>${snapshot.causal_node_count || 0}</strong></div>
        <div class="metric"><label>轮次</label><strong>${snapshot.valid_turn_count || 0}</strong></div>
        <div class="metric"><label>压缩率</label><strong>${fmt(snapshot.compression_ratio, 2)}</strong></div>
      `;
      const meta = document.getElementById("graphMeta");
      if (snapshot.initialized) {
        meta.textContent = `节省约 ${fmt((snapshot.estimated_space_saved || 0) * 100, 1)}% 空间`;
      } else {
        const turnCount = snapshot.valid_turn_count || 0;
        const threshold = snapshot.cold_start_threshold || 0;
        const currentDensity = snapshot.current_turn_density || 0;
        const avgDensity = snapshot.average_density || 0;
        const densityThreshold = snapshot.information_density_threshold || 0;
        const triggerReason = snapshot.trigger_reason;
        
        if (triggerReason) {
          meta.textContent = `信息密度 ${fmt(currentDensity * 100, 1)}% | 平均 ${fmt(avgDensity * 100, 1)}% | 触发: ${triggerReason}`;
        } else {
          meta.textContent = `冷启动 ${turnCount}/${threshold} | 信息密度 ${fmt(currentDensity * 100, 1)}% (阈值 ${fmt(densityThreshold * 100, 1)}%)`;
        }
      }
    }

    function tags(items) {
      return (items || []).map(item => `<span class="tag">${esc(item)}</span>`).join("");
    }

    function renderDetail(nodeData) {
      const box = document.getElementById("detailBox");
      const deleteBtn = document.getElementById("deleteNodeBtn");
      selectedNodeId = nodeData?.id || null;
      deleteBtn.style.display = currentView === "coarse" && selectedNodeId ? "inline-block" : "none";
      if (!nodeData) {
        box.innerHTML = `<div class="empty">未选择节点</div>`;
        return;
      }
      if (currentView === "coarse") {
        box.innerHTML = `
          <div class="record"><div class="record-title">${esc(nodeData.title || "未命名事件")}</div><div>${esc(nodeData.summary || "")}</div></div>
          <div class="record"><div class="record-title">视觉摘要</div><div>${esc(nodeData.visual_description || "无")}</div></div>
          <div class="record">
            <span class="tag">轮次 ${nodeData.turn_count || 0}</span>
            <span class="tag">实体 ${nodeData.entity_count || 0}</span>
            <span class="tag">时序 ${nodeData.temporal_count || 0}</span>
            <span class="tag">因果 ${nodeData.causal_count || 0}</span>
            <span class="tag">生存 ${fmt(nodeData.survival_weight, 2)}</span>
            <span class="tag">压缩 ${fmt(nodeData.compression_ratio, 2)}</span>
          </div>
          <div class="record"><div class="record-title">ID</div><div class="mono">${esc(nodeData.id)}</div></div>
          <div style="margin-top:12px;padding:8px;background:#e8f4f8;border-radius:4px;font-size:11px;color:#0c5460;">💡 双击节点可查看细粒度子图</div>
        `;
        return;
      }
      if (currentSubgraphTab === "entity") {
        box.innerHTML = `
          <div class="record"><div class="record-title">${esc(nodeData.name || "未命名实体")}</div><div>${tags([nodeData.entity_type])}</div></div>
          <div class="record"><div class="record-title">摘要</div><div>${esc(nodeData.summary || "")}</div></div>
          <div class="record">
            <span class="tag">频次 ${nodeData.frequency || 0}</span>
            <span class="tag">重要性 ${fmt(nodeData.importance, 2)}</span>
            <span class="tag">活跃 ${fmt(nodeData.active_weight, 2)}</span>
            <span class="tag">${nodeData.pinned ? "固化" : "可衰减"}</span>
          </div>
          <div class="record"><div class="record-title">ID</div><div class="mono">${esc(nodeData.id)}</div></div>
        `;
      } else if (currentSubgraphTab === "temporal") {
        box.innerHTML = `
          <div class="record"><div class="record-title">${esc(nodeData.stage_name || "未命名阶段")}</div><div>${tags([nodeData.node_type])}</div></div>
          <div class="record"><div class="record-title">摘要</div><div>${esc(nodeData.summary || "")}</div></div>
          <div class="record">
            <span class="tag">开始 ${nodeData.start_time ? new Date(nodeData.start_time * 1000).toLocaleTimeString() : '-'}</span>
            <span class="tag">结束 ${nodeData.end_time ? new Date(nodeData.end_time * 1000).toLocaleTimeString() : '-'}</span>
            <span class="tag">活跃 ${fmt(nodeData.active_weight, 2)}</span>
          </div>
          <div class="record"><div class="record-title">ID</div><div class="mono">${esc(nodeData.id)}</div></div>
        `;
      } else {
        box.innerHTML = `
          <div class="record"><div class="record-title">${esc(nodeData.text || "无")}</div><div>${tags([nodeData.node_type])}</div></div>
          <div class="record">
            <span class="tag">频次 ${nodeData.frequency || 0}</span>
            <span class="tag">重要性 ${fmt(nodeData.importance, 2)}</span>
            <span class="tag">活跃 ${fmt(nodeData.active_weight, 2)}</span>
            <span class="tag">${nodeData.pinned ? "固化" : "可衰减"}</span>
          </div>
          <div class="record"><div class="record-title">ID</div><div class="mono">${esc(nodeData.id)}</div></div>
        `;
      }
    }

    function renderRetrieval(retrieval, responseData) {
      const box = document.getElementById("retrievalBox");
      document.getElementById("answerState").textContent = responseData?.turn?.modality || "";
      if (!retrieval || !retrieval.steps || retrieval.steps.length === 0) {
        box.innerHTML = `<div class="empty">暂无路径</div>`;
        return;
      }
      let html = "";
      if (responseData?.answer) {
        html += `<div class="record"><div class="record-title">回答</div><div>${esc(responseData.answer)}</div></div>`;
      }
      if (retrieval.anchor) {
        const a = retrieval.anchor;
        html += `<div class="record"><div class="record-title">锚点事件：${esc(a.title)}</div><span class="tag">综合 ${fmt(a.score, 2)}</span></div>`;
      }
      retrieval.steps.forEach(step => {
        if (step.step === "multi_layer_retrieval") {
          html += `<div class="record"><div class="record-title">多层检索完成</div>
            <span class="tag">实体 ${step.entity_count || 0}</span>
            <span class="tag">时序 ${step.temporal_count || 0}</span>
            <span class="tag">因果 ${step.causal_count || 0}</span>
          </div>`;
        } else if (step.step === "no_fine_entry") {
          html += `<div class="record"><div class="record-title">${esc(step.message)}</div></div>`;
        }
      });
      if (retrieval.entity_results && retrieval.entity_results.length) {
        html += `<div class="record"><div class="record-title">实体匹配</div>${retrieval.entity_results.map(n => `<div class="muted">[${esc(n.entity_type)}] ${esc(n.name)} · ${fmt(n.score, 2)}</div>`).join("")}</div>`;
      }
      if (retrieval.causal_results && retrieval.causal_results.length) {
        html += `<div class="record"><div class="record-title">因果匹配</div>${retrieval.causal_results.map(n => `<div class="muted">[${esc(n.node_type)}] ${esc(n.text)} · ${fmt(n.score, 2)}</div>`).join("")}</div>`;
      }
      box.innerHTML = html;
    }

    function renderCoarseGraph() {
      currentView = "coarse";
      currentCoarseId = null;
      document.getElementById("subgraphTabs").style.display = "none";
      document.getElementById("backBtn").style.display = "none";
      document.getElementById("viewLabel").textContent = "粗粒度事件层";
      document.getElementById("graphTitle").textContent = "宏观态势";
      document.getElementById("legend").innerHTML = Object.entries({
        event: "#33658a"
      }).map(([key, color]) => `<span class="tag"><span class="swatch" style="background:${color}"></span>${key}</span>`).join("");

      const nodes = (currentGraph.events || []).map(node => ({
        id: node.id,
        label: node.title || node.summary || node.id,
        title: `${node.title || ""}\\n${node.summary || ""}`,
        value: 18 + (node.entity_count || 0) * 2 + (node.turn_count || 0),
        color: { background: "#e8f0f4", border: "#33658a" },
        font: { face: "Microsoft YaHei", color: "#20242a", size: 13 }
      }));
      const edges = (currentGraph.coarse_edges || []).map(edge => ({
        from: edge.source,
        to: edge.target,
        value: Math.max(1, (edge.score || 0.1) * 5),
        label: fmt(edge.score, 2),
        font: { align: "middle", size: 9 },
        color: { color: "#8d9898", opacity: 0.75 }
      }));
      drawNetwork(nodes, edges, { min: 16, max: 44 });
      
      // Only bind events once
      if (!network._eventsBound) {
        network._eventsBound = true;
        
        network.on("click", params => {
          if (!params.nodes.length) {
            renderDetail(null);
            return;
          }
          const nodeId = params.nodes[0];
          const node = (currentGraph.events || []).find(item => item.id === nodeId);
          renderDetail(node);
        });

        network.on("doubleClick", params => {
          if (!params.nodes.length) return;
          const nodeId = params.nodes[0];
          const sub = currentGraph.subgraphs?.[nodeId];
          if (sub && ((sub.entity_nodes?.length || 0) > 0 || (sub.temporal_nodes?.length || 0) > 0 || (sub.causal_nodes?.length || 0) > 0)) {
            currentCoarseId = nodeId;
            currentSubgraphTab = "entity";
            renderFineGraph(nodeId, "entity");
          }
        });
      }
    }

    function renderFineGraph(coarseId, tab) {
      currentView = "fine";
      currentSubgraphTab = tab;
      document.querySelectorAll(".subgraph-tab").forEach(t => {
        t.classList.toggle("active", t.dataset.tab === tab);
      });
      document.getElementById("subgraphTabs").style.display = "flex";
      document.getElementById("backBtn").style.display = "inline-block";
      document.getElementById("viewLabel").textContent = "细粒度子图";
      const topic = (currentGraph.events || []).find(item => item.id === coarseId);
      document.getElementById("graphTitle").textContent = topic?.title || "细粒度子图";

      const subgraph = currentGraph.subgraphs[coarseId] || {};
      let nodes = [], edges = [], colors = {};

      if (tab === "entity") {
        document.getElementById("legend").innerHTML = Object.entries(entityColors).map(([key, color]) => `<span class="tag"><span class="swatch" style="background:${color}"></span>${key}</span>`).join("");
        (subgraph.entity_nodes || []).forEach(node => {
          nodes.push({
            id: node.id,
            label: node.name && node.name.length > 18 ? node.name.slice(0, 18) + "..." : node.name,
            title: `[${node.entity_type}] ${node.summary}\\n重要性:${fmt(node.importance, 2)} 活跃:${fmt(node.active_weight, 2)}`,
            value: 10 + (node.frequency || 1) * 3 + (node.importance || 0) * 12,
            color: { background: entityColors[node.entity_type] || "#697684", border: "#20242a" },
            font: { face: "Microsoft YaHei", color: "#20242a", size: 12 }
          });
        });
        (subgraph.entity_edges || []).forEach(edge => {
          edges.push({
            from: edge.source_id,
            to: edge.target_id,
            value: Math.max(1, Math.min(edge.weight || 1, 8)),
            label: edge.relation_label || "",
            font: { align: "middle", size: 7 },
            color: { color: "#8d9898", opacity: 0.75 },
            arrows: "to"
          });
        });
      } else if (tab === "temporal") {
        document.getElementById("legend").innerHTML = Object.entries(temporalColors).map(([key, color]) => `<span class="tag"><span class="swatch" style="background:${color}"></span>${key}</span>`).join("");
        (subgraph.temporal_nodes || []).forEach(node => {
          nodes.push({
            id: node.id,
            label: node.stage_name || node.node_type,
            title: `[${node.node_type}] ${node.summary}\\n${node.start_time ? new Date(node.start_time*1000).toLocaleTimeString() : '-'} ~ ${node.end_time ? new Date(node.end_time*1000).toLocaleTimeString() : '-'}`,
            value: 10 + (node.frequency || 1) * 3,
            color: { background: temporalColors[node.node_type] || "#697684", border: "#20242a" },
            font: { face: "Microsoft YaHei", color: "#20242a", size: 12 }
          });
        });
        (subgraph.temporal_edges || []).forEach(edge => {
          edges.push({
            from: edge.source_id,
            to: edge.target_id,
            value: 2,
            label: edge.relation_label || "",
            font: { align: "middle", size: 7 },
            color: { color: "#33658a", opacity: 0.75 },
            arrows: "to"
          });
        });
      } else {
        document.getElementById("legend").innerHTML = Object.entries(causalColors).map(([key, color]) => `<span class="tag"><span class="swatch" style="background:${color}"></span>${key}</span>`).join("");
        (subgraph.causal_nodes || []).forEach(node => {
          nodes.push({
            id: node.id,
            label: node.text && node.text.length > 18 ? node.text.slice(0, 18) + "..." : node.text,
            title: `[${node.node_type}] ${node.text}\\n重要性:${fmt(node.importance, 2)} 活跃:${fmt(node.active_weight, 2)}`,
            value: 10 + (node.frequency || 1) * 3 + (node.importance || 0) * 12,
            color: { background: causalColors[node.node_type] || "#697684", border: "#20242a" },
            font: { face: "Microsoft YaHei", color: "#20242a", size: 12 }
          });
        });
        (subgraph.causal_edges || []).forEach(edge => {
          edges.push({
            from: edge.source_id,
            to: edge.target_id,
            value: Math.max(1, Math.min(edge.weight || 1, 8)),
            label: edge.relation_label || "",
            font: { align: "middle", size: 7 },
            color: { color: "#b9422f", opacity: 0.75 },
            arrows: "to"
          });
        });
      }
      drawNetwork(nodes, edges, { min: 9, max: 32 });
      network?.on("click", params => {
        if (!params.nodes.length) {
          renderDetail(null);
          return;
        }
        const nodeId = params.nodes[0];
        let node = null;
        if (tab === "entity") node = (subgraph.entity_nodes || []).find(item => item.id === nodeId);
        else if (tab === "temporal") node = (subgraph.temporal_nodes || []).find(item => item.id === nodeId);
        else node = (subgraph.causal_nodes || []).find(item => item.id === nodeId);
        renderDetail(node);
      });
    }

    function drawNetwork(nodes, edges, scaling) {
      const container = document.getElementById("graph");
      if (!window.vis) {
        container.innerHTML = `<div class="empty">图渲染库未加载</div>`;
        return;
      }
      if (network) network.destroy();
      network = new vis.Network(container, { nodes, edges }, {
        layout: { improvedLayout: true },
        physics: {
          stabilization: false,
          barnesHut: { gravitationalConstant: -3600, springLength: 155, damping: 0.24 }
        },
        interaction: { hover: true, navigationButtons: false },
        nodes: { shape: "dot", scaling }
      });
    }

    async function loadGraph() {
      const response = await fetch("/api/graph");
      const raw = await response.json();
      currentGraph = {
        events: raw.events || [],
        coarse_edges: raw.coarse_edges || [],
        subgraphs: raw.subgraphs || {},
        initialized: raw.initialized || false
      };
      renderMetrics(raw);
      if (currentView === "fine" && currentCoarseId) {
        renderFineGraph(currentCoarseId, currentSubgraphTab || "entity");
      } else {
        renderCoarseGraph();
      }
      if (!selectedNodeId) renderDetail(null);
      loadHistory();
    }

    async function sendChat() {
      const questionInput = document.getElementById("question");
      const question = questionInput.value.trim();
      if (!question) return;
      const payload = {
        question,
        image_paths: [],
        visual_description: ""
      };
      const loading = document.getElementById("loading");
      loading.style.display = "block";
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        renderRetrieval(data.retrieval || {}, data);
        // Always reload full graph data including subgraphs after chat
        currentGraph.events = data.graph?.events || currentGraph.events;
        currentGraph.coarse_edges = data.graph?.coarse_edges || [];
        currentGraph.initialized = data.graph?.initialized || false;
        // Reload full subgraph data from /api/graph to ensure completeness
        try {
          const graphResp = await fetch("/api/graph");
          const fullGraph = await graphResp.json();
          currentGraph.subgraphs = fullGraph.subgraphs || {};
        } catch (e) {
          currentGraph.subgraphs = data.graph?.subgraphs || currentGraph.subgraphs;
        }
        renderMetrics(data.graph || {});
        renderCoarseGraph();
        if (currentView === "fine" && currentCoarseId) {
          renderFineGraph(currentCoarseId, currentSubgraphTab);
        }
        // Show changes
        renderChanges(data.changes || []);
        // Clear input after successful send
        questionInput.value = "";
      } finally {
        loading.style.display = "none";
      }
    }

    async function seedGraph() {
      const response = await fetch("/api/seed", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const data = await response.json();
      renderRetrieval({ steps: [{ step: "multi_layer_retrieval", entity_count: data.entity_nodes || 0, temporal_count: data.temporal_nodes || 0, causal_count: data.causal_nodes || 0 }] }, { answer: `已写入 ${data.inserted || 0} 条样例` });
      await loadGraph();
    }

    async function deleteSelectedNode() {
      if (!selectedNodeId) return;
      if (!confirm(`删除事件 ${selectedNodeId}？`)) return;
      const response = await fetch(`/api/nodes/${encodeURIComponent(selectedNodeId)}`, { method: "DELETE" });
      const data = await response.json();
      currentGraph.events = data.graph?.events || [];
      currentGraph.coarse_edges = data.graph?.coarse_edges || [];
      selectedNodeId = null;
      renderMetrics(data.graph || {});
      renderCoarseGraph();
      renderDetail(null);
    }

    function renderHistory(turns) {
      const box = document.getElementById("historyBox");
      if (!turns || turns.length === 0) {
        box.innerHTML = '<div class="muted" style="padding:8px;">暂无对话记录</div>';
        return;
      }
      
      const sorted = turns.slice().sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
      box.innerHTML = sorted.map(turn => {
        const time = turn.timestamp ? new Date(turn.timestamp * 1000).toLocaleString('zh-CN') : '-';
        const valid = turn.valid ? '[OK]' : '[无效]';
        const eventId = turn.event_id ? turn.event_id.substring(0, 12) : '-';
        const density = turn.information_density ? `(密度: ${(turn.information_density * 100).toFixed(1)}%)` : '';
        return `
          <div style="padding:10px;border-bottom:1px solid #e8ecf0;position:relative;">
            <button style="position:absolute;top:8px;right:8px;background:#e74c3c;color:#fff;border:none;border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;" onclick="event.stopPropagation();deleteTurn('${turn.id}')">删除</button>
            <div style="font-size:11px;color:#8d9898;margin-bottom:4px;cursor:pointer;padding-right:50px;" onclick="showTurnDetail(${JSON.stringify(turn).replace(/"/g, '&quot;')})">
              ${time} ${valid} 事件:${eventId} ${density}
            </div>
            <div style="font-size:12px;margin-bottom:4px;color:#20242a;cursor:pointer;padding-right:50px;" onclick="showTurnDetail(${JSON.stringify(turn).replace(/"/g, '&quot;')})"><strong>Q:</strong> ${esc(turn.question || '-')}</div>
            <div style="font-size:12px;color:#5a6868;cursor:pointer;padding-right:50px;" onclick="showTurnDetail(${JSON.stringify(turn).replace(/"/g, '&quot;')})"><strong>A:</strong> ${esc((turn.answer || '').substring(0, 100))}${(turn.answer || '').length > 100 ? '...' : ''}</div>
          </div>
        `;
      }).join('');
    }

    window.showTurnDetail = function(turn) {
      const box = document.getElementById("detailBox");
      const time = turn.timestamp ? new Date(turn.timestamp * 1000).toLocaleString('zh-CN') : '-';
      const density = turn.information_density ? `${(turn.information_density * 100).toFixed(1)}%` : '-';
      box.innerHTML = `
        <div style="padding:10px;">
          <div style="font-size:11px;color:#8d9898;margin-bottom:8px;">
            时间: ${time}<br/>
            状态: ${turn.valid ? '有效' : '无效'} | 模态: ${turn.modality || 'text'}<br/>
            信息密度: ${density} | 事件: ${turn.event_id || '-'}
          </div>
          <div style="margin-bottom:8px;">
            <div style="font-size:11px;color:#8d9898;margin-bottom:4px;">问题:</div>
            <div style="font-size:12px;color:#20242a;padding:6px;background:#f5f7fa;border-radius:4px;">${esc(turn.question || '-')}</div>
          </div>
          <div style="margin-bottom:8px;">
            <div style="font-size:11px;color:#8d9898;margin-bottom:4px;">回答:</div>
            <div style="font-size:12px;color:#20242a;padding:6px;background:#f5f7fa;border-radius:4px;">${esc(turn.answer || '-')}</div>
          </div>
          ${turn.local_summary ? `
          <div>
            <div style="font-size:11px;color:#8d9898;margin-bottom:4px;">摘要:</div>
            <div style="font-size:12px;color:#5a6868;padding:6px;background:#fafbfc;border-radius:4px;">${esc(turn.local_summary)}</div>
          </div>
          ` : ''}
        </div>
      `;
    };

    window.viewSubgraph = function(eventId) {
      currentCoarseId = eventId;
      currentSubgraphTab = "entity";
      renderFineGraph(eventId, "entity");
    };

    window.deleteTurn = async function(turnId) {
      if (!confirm(`确定删除此对话？相关节点将自动更新。`)) return;
      try {
        const response = await fetch(`/api/turns/${encodeURIComponent(turnId)}`, { method: "DELETE" });
        const data = await response.json();
        if (data.deleted) {
          currentGraph.events = data.graph?.events || [];
          currentGraph.coarse_edges = data.graph?.coarse_edges || [];
          currentGraph.subgraphs = data.graph?.subgraphs || {};
          currentGraph.initialized = data.graph?.initialized || false;
          renderMetrics(data.graph || {});
          renderCoarseGraph();
          if (currentView === "fine" && currentCoarseId) {
            renderFineGraph(currentCoarseId, currentSubgraphTab);
          }
          loadHistory();
        } else {
          alert("删除失败");
        }
      } catch (e) {
        alert("删除失败: " + e.message);
      }
    };

    async function loadHistory() {
      try {
        const response = await fetch("/api/graph");
        const data = await response.json();
        renderHistory(data.raw_turns || []);
      } catch (e) {
        console.error("加载历史对话失败:", e);
      }
    }

    function renderChanges(changes) {
      const panel = document.getElementById("changesPanel");
      const box = document.getElementById("changesBox");
      if (!changes || changes.length === 0) {
        panel.style.display = "none";
        return;
      }
      panel.style.display = "block";
      const typeIcons = {
        "event_created": "⭐",
        "event_updated": "📌",
        "entity_created": "🔵",
        "temporal_created": "🟢",
        "causal_created": "🔴",
        "causal_corrected": "✅",
        "entity_corrected": "✅",
        "causal_replaced": "🔄",
        "nodes_deleted": "🗑️",
        "causal_conflict_detected": "⚠️",
      };
      box.innerHTML = changes.map(change => {
        const icon = typeIcons[change.event_type] || "📋";
        return `
          <div style="margin-bottom:6px; padding:8px; background:#232529; border-radius:6px; border-left:3px solid #33658a;">
            <div style="font-size:13px; color:#e1e4e8;">${icon} ${change.message || ""}</div>
          </div>
        `;
      }).join("");
    }

    function hideChanges() {
      document.getElementById("changesPanel").style.display = "none";
    }

    document.getElementById("sendBtn").addEventListener("click", sendChat);
    document.getElementById("seedBtn").addEventListener("click", seedGraph);
    document.getElementById("deleteNodeBtn").addEventListener("click", deleteSelectedNode);
    document.getElementById("refreshHistoryBtn").addEventListener("click", loadHistory);
    document.getElementById("backBtn").addEventListener("click", () => {
      currentView = "coarse";
      renderCoarseGraph();
      renderDetail(null);
    });
    document.getElementById("subgraphTabs").addEventListener("click", e => {
      if (!currentCoarseId) return;
      const tab = e.target.closest(".subgraph-tab");
      if (!tab) return;
      currentSubgraphTab = tab.dataset.tab;
      renderFineGraph(currentCoarseId, currentSubgraphTab);
    });
    loadGraph();
  </script>
</body>
</html>"""
