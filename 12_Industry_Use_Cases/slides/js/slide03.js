/* ══════════════════════════════════════════════════════════════
   SLIDE 3: AGENT CONSTELLATION (slide index 2)
   ══════════════════════════════════════════════════════════════ */

function buildConstellation() {
  const svg = document.getElementById('constellation-svg');
  if (svg.childNodes.length > 0) return;

  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  const cx = 300, cy = 200, cr = 50;
  const agents = [
    { label: 'Model Selection', desc: 'Picks the best algorithm for your data', color: '#f97316', angle: -30, icon: '\u2699' },
    { label: 'Dataset Analyzer', desc: 'Detects your data format automatically', color: '#818cf8', angle: 210, icon: '\ud83d\udd0d' },
    { label: 'Error Investigation', desc: 'Diagnoses failures & suggests fixes', color: '#ef4444', angle: 90, icon: '\ud83d\udee1' },
  ];

  // Center circle with pulse
  const centerPulse = el('circle', { cx: cx, cy: cy, r: '55', fill: 'none', stroke: '#e2e8f0', 'stroke-width': '1', opacity: '0.3' });
  const pulseAnim = el('animate', { attributeName: 'r', values: '50;60;50', dur: '3s', repeatCount: 'indefinite' });
  centerPulse.appendChild(pulseAnim);
  const pulseOpac = el('animate', { attributeName: 'opacity', values: '0.3;0.1;0.3', dur: '3s', repeatCount: 'indefinite' });
  centerPulse.appendChild(pulseOpac);
  svg.appendChild(centerPulse);

  const centerCircle = el('circle', { cx: cx, cy: cy, r: cr, fill: 'rgba(226,232,240,0.08)', stroke: '#e2e8f0', 'stroke-width': '2' });
  svg.appendChild(centerCircle);
  const centerLabel = el('text', { x: cx, y: cy + 5, 'text-anchor': 'middle', fill: '#e2e8f0', 'font-size': '14', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
  centerLabel.textContent = 'Your Data';
  svg.appendChild(centerLabel);

  // Agent nodes
  const agentRadius = 160;
  agents.forEach((a, i) => {
    const rad = a.angle * Math.PI / 180;
    const ax = cx + Math.cos(rad) * agentRadius;
    const ay = cy + Math.sin(rad) * agentRadius;

    // Connection line (dashed)
    const line = el('line', { x1: cx, y1: cy, x2: ax, y2: ay, stroke: a.color, 'stroke-width': '1.5', 'stroke-dasharray': '6 4', opacity: '0.4', class: 'const-line' });
    svg.appendChild(line);

    // Flow dot along the line
    const flowDot = el('circle', { r: '3', fill: a.color, class: 'flow-dot' });
    const flowMotion = el('animateMotion', { dur: '2s', repeatCount: 'indefinite', begin: `${i * 0.6}s`, path: `M${cx},${cy} L${ax},${ay}` });
    // Use relative path
    const fmPath = el('animateMotion', { dur: '2s', repeatCount: 'indefinite', begin: `${i * 0.6}s`, path: `M0,0 L${ax-cx},${ay-cy}` });
    flowDot.setAttribute('cx', String(cx));
    flowDot.setAttribute('cy', String(cy));
    flowDot.appendChild(fmPath);
    svg.appendChild(flowDot);

    // Agent group
    const g = el('g', { class: 'const-agent', id: `const-agent-${i}`, style: `transform-origin: ${ax}px ${ay}px` });

    const agentCircle = el('circle', { cx: ax, cy: ay, r: '36', fill: a.color.replace(')', ',0.1)').replace('#', 'rgba(').replace(/([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/, (m,r,g,b) => `${parseInt(r,16)},${parseInt(g,16)},${parseInt(b,16)}`), stroke: a.color, 'stroke-width': '2' });
    // Simpler fill approach
    agentCircle.setAttribute('fill', 'rgba(0,0,0,0.3)');
    g.appendChild(agentCircle);

    const agentIcon = el('text', { x: ax, y: ay - 2, 'text-anchor': 'middle', fill: a.color, 'font-size': '18' });
    agentIcon.textContent = a.icon;
    g.appendChild(agentIcon);

    const agentLabel = el('text', { x: ax, y: String(parseFloat(ay) + 52), 'text-anchor': 'middle', fill: a.color, 'font-size': '11', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
    agentLabel.textContent = a.label;
    g.appendChild(agentLabel);

    const agentDesc = el('text', { x: ax, y: String(parseFloat(ay) + 66), 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '9', 'font-weight': '400', 'font-family': 'Inter, sans-serif' });
    agentDesc.textContent = a.desc;
    g.appendChild(agentDesc);

    svg.appendChild(g);
  });
}

let constellationPlayed = false;
function playConstellation() {
  if (constellationPlayed) return;
  constellationPlayed = true;
  [0, 1, 2].forEach((i) => {
    setTimeout(() => {
      document.getElementById(`const-agent-${i}`)?.classList.add('visible');
    }, 400 + i * 500);
  });
}

registerAnim(2,
  function enter() { buildConstellation(); playConstellation(); },
  null
);
