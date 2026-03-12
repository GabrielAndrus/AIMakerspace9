/* ══════════════════════════════════════════════════════════════
   SLIDE 7: D3.js STATE MACHINE (slide index 6)
   ══════════════════════════════════════════════════════════════ */

let d3GraphBuilt = false;
let graphTokenAnim = null;

const GRAPH_NODES = [
  { id: 'query',    label: 'Create Search Queries',  sub: 'Turn error message into searches', color: '#f97316', x: 140, y: 50,  w: 190, h: 50 },
  { id: 'search',   label: 'Search the Web',          sub: 'GitHub, StackOverflow, HuggingFace', color: '#10b981', x: 140, y: 140, w: 190, h: 50 },
  { id: 'analyzer', label: 'Rate the Results',        sub: 'Are these results actually useful?', color: '#818cf8', x: 140, y: 230, w: 190, h: 50 },
  { id: 'gate',     label: 'Good enough?',             color: '#f59e0b', x: 160, y: 320, w: 150, h: 40, diamond: true },
  { id: 'fetcher',  label: 'Read Full Pages',          sub: 'Fetch top 3 matching documents', color: '#22d3ee', x: 70,  y: 400, w: 170, h: 50 },
  { id: 'synth',    label: 'Write Fix Instructions',   sub: 'Root cause + step-by-step solution', color: '#f59e0b', x: 70,  y: 490, w: 190, h: 50 },
];

const GRAPH_EDGES = [
  { from: 'query',    to: 'search' },
  { from: 'search',   to: 'analyzer' },
  { from: 'analyzer', to: 'gate' },
  { from: 'gate',     to: 'fetcher',  label: '≥ 0.5', color: '#10b981' },
  { from: 'gate',     to: 'query',    label: '< 0.5', color: '#ef4444', feedback: true },
  { from: 'fetcher',  to: 'synth' },
];

function buildD3Graph() {
  if (d3GraphBuilt) return;
  d3GraphBuilt = true;

  const container = document.getElementById('d3-graph-container');
  const svgW = 520, svgH = 570;
  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${svgW} ${svgH}`)
    .attr('width', '100%');

  // Arrowhead marker
  svg.append('defs').append('marker')
    .attr('id', 'arrowhead').attr('viewBox', '0 0 10 7')
    .attr('refX', 10).attr('refY', 3.5)
    .attr('markerWidth', 8).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('polygon').attr('points', '0 0, 10 3.5, 0 7').attr('fill', '#3d4760');

  svg.append('defs').append('marker')
    .attr('id', 'arrowhead-red').attr('viewBox', '0 0 10 7')
    .attr('refX', 10).attr('refY', 3.5)
    .attr('markerWidth', 8).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('polygon').attr('points', '0 0, 10 3.5, 0 7').attr('fill', '#ef4444');

  svg.append('defs').append('marker')
    .attr('id', 'arrowhead-green').attr('viewBox', '0 0 10 7')
    .attr('refX', 10).attr('refY', 3.5)
    .attr('markerWidth', 8).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('polygon').attr('points', '0 0, 10 3.5, 0 7').attr('fill', '#10b981');

  // Glow filter
  const filter = svg.append('defs').append('filter').attr('id', 'glow');
  filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
  filter.append('feMerge').selectAll('feMergeNode')
    .data(['blur', 'SourceGraphic']).enter()
    .append('feMergeNode').attr('in', d => d);

  const nodeMap = {};
  GRAPH_NODES.forEach(n => nodeMap[n.id] = n);

  // Draw edges
  GRAPH_EDGES.forEach(e => {
    const from = nodeMap[e.from], to = nodeMap[e.to];
    const x1 = from.x + from.w / 2, y1 = from.y + from.h;
    const x2 = to.x + to.w / 2, y2 = to.y;

    if (e.feedback) {
      // Curved feedback arc on the right
      const path = svg.append('path')
        .attr('d', `M${x1 + from.w/2 - 10},${from.y + from.h/2} C${svgW - 40},${from.y + from.h/2} ${svgW - 40},${to.y + to.h/2} ${x2 + to.w/2 - 10},${to.y + to.h/2}`)
        .attr('class', 'd3-edge d3-edge-feedback')
        .attr('marker-end', 'url(#arrowhead-red)')
        .attr('id', `edge-${e.from}-${e.to}`);

      // Label
      svg.append('text')
        .attr('x', svgW - 30).attr('y', (from.y + to.y) / 2 + 60)
        .attr('fill', e.color).attr('font-size', '11').attr('font-weight', '600')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(e.label);

      // "Refine" label
      svg.append('text')
        .attr('x', svgW - 60).attr('y', (from.y + to.y) / 2 + 80)
        .attr('fill', '#ef4444').attr('font-size', '10').attr('font-weight', '600')
        .attr('font-family', 'Inter, sans-serif')
        .text('Refine & Retry');
    } else {
      const markerUrl = e.color === '#10b981' ? 'url(#arrowhead-green)' : 'url(#arrowhead)';
      svg.append('line')
        .attr('x1', x1).attr('y1', y1)
        .attr('x2', x2).attr('y2', y2)
        .attr('class', 'd3-edge')
        .attr('stroke', e.color || '#3d4760')
        .attr('marker-end', markerUrl)
        .attr('id', `edge-${e.from}-${e.to}`);

      if (e.label) {
        svg.append('text')
          .attr('x', (x1 + x2) / 2 - 40).attr('y', (y1 + y2) / 2 + 4)
          .attr('fill', e.color).attr('font-size', '10').attr('font-weight', '600')
          .attr('font-family', 'JetBrains Mono, monospace')
          .text(e.label);
      }
    }
  });

  // Draw nodes
  GRAPH_NODES.forEach(n => {
    const g = svg.append('g').attr('class', 'd3-node').attr('id', `node-${n.id}`);

    if (n.diamond) {
      // Diamond shape
      const cx = n.x + n.w / 2, cy = n.y + n.h / 2;
      g.append('polygon')
        .attr('points', `${cx},${cy-n.h/2} ${cx+n.w/2},${cy} ${cx},${cy+n.h/2} ${cx-n.w/2},${cy}`)
        .attr('fill', 'var(--surface2)')
        .attr('stroke', n.color)
        .attr('stroke-width', 2);
      g.append('text')
        .attr('x', cx).attr('y', cy + 4)
        .attr('text-anchor', 'middle')
        .attr('fill', n.color).attr('font-size', '12').attr('font-weight', '700')
        .text(n.label);
    } else {
      g.append('rect')
        .attr('x', n.x).attr('y', n.y)
        .attr('width', n.w).attr('height', n.h)
        .attr('stroke', n.color);
      g.append('text')
        .attr('x', n.x + n.w / 2).attr('y', n.y + (n.sub ? n.h / 2 - 1 : n.h / 2 + 5))
        .attr('text-anchor', 'middle')
        .attr('fill', n.color)
        .text(n.label);
      if (n.sub) {
        g.append('text')
          .attr('x', n.x + n.w / 2).attr('y', n.y + n.h / 2 + 13)
          .attr('text-anchor', 'middle')
          .attr('fill', '#94a3b8')
          .attr('font-size', '9')
          .text(n.sub);
      }
    }
  });

  // Token (hidden initially)
  svg.append('circle')
    .attr('class', 'd3-token')
    .attr('id', 'graph-token')
    .attr('cx', -20).attr('cy', -20)
    .attr('r', 7)
    .attr('filter', 'url(#glow)');
}

function animateGraphToken(retryPath) {
  const nodeMap = {};
  GRAPH_NODES.forEach(n => { nodeMap[n.id] = { cx: n.x + n.w / 2, cy: n.y + n.h / 2 }; });

  const token = d3.select('#graph-token');
  const iterEl = document.getElementById('graph-iter');

  // Path: normal or retry
  let path;
  if (retryPath) {
    path = ['query', 'search', 'analyzer', 'gate', 'query', 'search', 'analyzer', 'gate', 'fetcher', 'synth'];
  } else {
    path = ['query', 'search', 'analyzer', 'gate', 'fetcher', 'synth'];
  }

  // Cancel previous
  if (graphTokenAnim) { graphTokenAnim.cancelled = true; }
  const state = { cancelled: false };
  graphTokenAnim = state;

  let step = 0;
  let iteration = 1;
  function next() {
    if (state.cancelled || step >= path.length) return;
    const nodeId = path[step];
    const pos = nodeMap[nodeId];

    // Track iteration
    if (retryPath && step === 4) { iteration = 2; iterEl.textContent = `Iteration: 2 / 3`; }
    if (!retryPath) iterEl.textContent = `Iteration: 1 / 3`;

    // Glow the node
    const nodeEl = document.getElementById(`node-${nodeId}`);
    if (nodeEl) {
      nodeEl.querySelector('rect,polygon').style.filter = 'drop-shadow(0 0 8px ' + (GRAPH_NODES.find(n=>n.id===nodeId)?.color || '#f97316') + ')';
      setTimeout(() => {
        if (nodeEl.querySelector('rect,polygon'))
          nodeEl.querySelector('rect,polygon').style.filter = 'none';
      }, 800);
    }

    token.transition().duration(500).ease(d3.easeCubicInOut)
      .attr('cx', pos.cx).attr('cy', pos.cy)
      .on('end', () => {
        step++;
        setTimeout(next, 400);
      });
  }

  // Start
  const startPos = nodeMap[path[0]];
  token.attr('cx', startPos.cx).attr('cy', startPos.cy - 40);
  setTimeout(next, 300);
}

function replayGraphToken(retry) {
  animateGraphToken(retry);
}

registerAnim(6,
  function enter() {
    buildD3Graph();
    setTimeout(() => animateGraphToken(false), 500);
  },
  function leave() {
    if (graphTokenAnim) { graphTokenAnim.cancelled = true; graphTokenAnim = null; }
  }
);
