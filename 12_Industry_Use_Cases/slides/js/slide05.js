/* ══════════════════════════════════════════════════════════════
   SLIDE 5: ANIMATED PIPELINE (slide index 4)
   ══════════════════════════════════════════════════════════════ */

function buildPipeline() {
  const svg = document.getElementById('pipeline-svg');
  if (svg.childNodes.length > 0) return;

  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  const nodes = [
    { label: 'Study Data',        sub: 'What kind of problem is this?', icon: '\ud83d\udd0d', color: '#f97316', x: 100 },
    { label: 'Make Searchable',   sub: 'Convert data traits to numbers', icon: '\u27a1',       color: '#10b981', x: 280 },
    { label: 'Find Similar Docs', sub: 'Search sklearn documentation',   icon: '\ud83d\uddc4', color: '#22d3ee', x: 460 },
    { label: 'AI Picks Models',   sub: 'LLM reasons over retrieved docs', icon: '\ud83e\udde0', color: '#818cf8', x: 640 },
    { label: 'Build & Test',      sub: 'Train ensemble, evaluate, export', icon: '\ud83d\udcc8', color: '#f59e0b', x: 820 },
  ];

  const y = 100, r = 32;

  // Connection lines
  for (let i = 0; i < nodes.length - 1; i++) {
    const line = el('line', {
      x1: String(nodes[i].x + r + 4), y1: y,
      x2: String(nodes[i+1].x - r - 4), y2: y,
      stroke: '#2e3648', 'stroke-width': '2'
    });
    svg.appendChild(line);
  }

  // Nodes
  nodes.forEach((n, i) => {
    const g = el('g', { class: 'pipeline-node', id: `pipe-node-${i}` });

    const circle = el('circle', { cx: n.x, cy: y, r: r, fill: 'var(--surface)', stroke: n.color, 'stroke-width': '2' });
    g.appendChild(circle);

    const icon = el('text', { x: n.x, y: y - 2, 'text-anchor': 'middle', fill: n.color, 'font-size': '18' });
    icon.textContent = n.icon;
    g.appendChild(icon);

    const label = el('text', { x: n.x, y: String(y + r + 22), 'text-anchor': 'middle', fill: n.color, 'font-size': '12', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
    label.textContent = n.label;
    g.appendChild(label);

    const sub = el('text', { x: n.x, y: String(y + r + 36), 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '9', 'font-family': 'Inter, sans-serif' });
    sub.textContent = n.sub;
    g.appendChild(sub);

    svg.appendChild(g);
  });

  // Packet label that transforms
  const packetLabels = ['Raw CSV', 'Data Profile', 'Search Query', 'Matching Docs', 'Final Models'];
  const packetText = el('text', { id: 'pipe-packet-label', x: '100', y: String(y - r - 14), 'text-anchor': 'middle', fill: '#f97316', 'font-size': '11', 'font-weight': '700', 'font-family': 'JetBrains Mono, monospace', opacity: '0' });
  packetText.textContent = 'CSV';
  svg.appendChild(packetText);

  // Packet dot
  const packet = el('circle', { id: 'pipe-packet', cx: '100', cy: y, r: '8', fill: '#f97316', opacity: '0' });
  svg.appendChild(packet);

  // Vector scatter group (hidden, appears at node 2/Search)
  const scatterG = el('g', { class: 'vector-scatter', id: 'vector-scatter' });
  const scatterBg = el('rect', { x: '370', y: '170', width: '180', height: '90', rx: '8', fill: 'rgba(34,211,238,0.05)', stroke: 'rgba(34,211,238,0.2)', 'stroke-width': '1' });
  scatterG.appendChild(scatterBg);

  // Scatter label
  const scatterLabel = el('text', { x: '460', y: '167', 'text-anchor': 'middle', fill: '#22d3ee', 'font-size': '9', 'font-weight': '600', 'font-family': 'Inter, sans-serif' });
  scatterLabel.textContent = '5 most relevant docs found';
  scatterG.appendChild(scatterLabel);

  // Random dots representing vectors
  const dots = [[390,200],[410,220],[430,195],[450,230],[470,210],[380,240],[420,245],[460,195],[440,215],[490,225],
                [400,205],[425,235],[445,200],[455,220],[480,215],[395,225],[415,210],[435,240],[465,205],[485,230]];
  const highlights = [2, 4, 8, 13, 18]; // nearest neighbors
  dots.forEach((d, i) => {
    const dot = el('circle', { cx: d[0], cy: d[1], r: highlights.includes(i) ? '4' : '2.5',
      fill: highlights.includes(i) ? '#22d3ee' : '#64748b', opacity: highlights.includes(i) ? '0.9' : '0.4' });
    scatterG.appendChild(dot);
  });
  // Query point
  const qPoint = el('circle', { cx: '440', cy: '215', r: '5', fill: '#f97316', stroke: '#f97316', 'stroke-width': '1' });
  scatterG.appendChild(qPoint);
  // Lines to highlights
  highlights.forEach(hi => {
    const line = el('line', { x1: '440', y1: '215', x2: dots[hi][0], y2: dots[hi][1], stroke: '#22d3ee', 'stroke-width': '1', opacity: '0.5' });
    scatterG.appendChild(line);
  });

  svg.appendChild(scatterG);
}

let pipelinePlayed = false;
function playPipeline() {
  if (pipelinePlayed) return;
  pipelinePlayed = true;

  const nodes = [100, 280, 460, 640, 820];
  const labels = ['Raw CSV', 'Data Profile', 'Search Query', 'Matching Docs', 'Final Models'];
  const packet = document.getElementById('pipe-packet');
  const packetLabel = document.getElementById('pipe-packet-label');

  packet?.setAttribute('opacity', '0.9');
  packetLabel?.setAttribute('opacity', '1');

  let step = 0;
  function nextStep() {
    if (step >= nodes.length) return;

    // Activate node
    const node = document.getElementById(`pipe-node-${step}`);
    node?.classList.add('active');

    // Move packet
    packet?.setAttribute('cx', String(nodes[step]));
    packetLabel?.setAttribute('x', String(nodes[step]));
    packetLabel.textContent = labels[step];

    // Show vector scatter at Search node
    if (step === 2) {
      setTimeout(() => document.getElementById('vector-scatter')?.classList.add('visible'), 300);
    }
    // Show prompt snippet at Reason node
    if (step === 3) {
      setTimeout(() => document.getElementById('pipeline-prompt')?.classList.add('visible'), 300);
    }

    // Deactivate previous
    if (step > 0) {
      setTimeout(() => document.getElementById(`pipe-node-${step-1}`)?.classList.remove('active'), 600);
    }

    step++;
    if (step < nodes.length) setTimeout(nextStep, 800);
  }

  setTimeout(nextStep, 400);
}

registerAnim(4,
  function enter() { buildPipeline(); playPipeline(); },
  null
);
