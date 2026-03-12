/* ══════════════════════════════════════════════════════════════
   SLIDE 4: ISOMETRIC ARCHITECTURE + BOOT SEQUENCE
   ══════════════════════════════════════════════════════════════ */

function buildIsometricSVG() {
  const svg = document.getElementById('iso-svg');
  // Helper: draw an isometric box
  function isoBox(cx, cy, w, h, d, fill, stroke, label, port) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

    // Top face
    const top = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    top.setAttribute('points',
      `${cx},${cy - d} ${cx + w / 2},${cy - d + h * 0.3} ${cx},${cy - d + h * 0.6} ${cx - w / 2},${cy - d + h * 0.3}`);
    top.setAttribute('fill', fill);
    top.setAttribute('fill-opacity', '0.3');
    top.setAttribute('stroke', stroke);
    top.setAttribute('stroke-width', '1.5');

    // Left face
    const left = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    left.setAttribute('points',
      `${cx - w / 2},${cy - d + h * 0.3} ${cx},${cy - d + h * 0.6} ${cx},${cy + h * 0.6 - d + d} ${cx - w / 2},${cy + h * 0.3 - d + d}`);
    left.setAttribute('fill', fill);
    left.setAttribute('fill-opacity', '0.15');
    left.setAttribute('stroke', stroke);
    left.setAttribute('stroke-width', '1');

    // Right face
    const right = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    right.setAttribute('points',
      `${cx + w / 2},${cy - d + h * 0.3} ${cx},${cy - d + h * 0.6} ${cx},${cy + h * 0.6 - d + d} ${cx + w / 2},${cy + h * 0.3 - d + d}`);
    right.setAttribute('fill', fill);
    right.setAttribute('fill-opacity', '0.08');
    right.setAttribute('stroke', stroke);
    right.setAttribute('stroke-width', '1');

    g.appendChild(left);
    g.appendChild(right);
    g.appendChild(top);

    // Label
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', cx);
    text.setAttribute('y', cy - d + h * 0.25);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', stroke);
    text.setAttribute('font-family', 'Inter, sans-serif');
    text.setAttribute('font-weight', '700');
    text.setAttribute('font-size', '13');
    text.textContent = label;
    g.appendChild(text);

    // Port
    if (port) {
      const pText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      pText.setAttribute('x', cx);
      pText.setAttribute('y', cy - d + h * 0.25 + 16);
      pText.setAttribute('text-anchor', 'middle');
      pText.setAttribute('fill', '#64748b');
      pText.setAttribute('font-family', 'JetBrains Mono, monospace');
      pText.setAttribute('font-size', '10');
      pText.textContent = port;
      g.appendChild(pText);
    }

    svg.appendChild(g);
    return g;
  }

  // Connection dots path helper
  function addFlowDots(x1, y1, x2, y2, color, delayMs) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x1); line.setAttribute('y1', y1);
    line.setAttribute('x2', x2); line.setAttribute('y2', y2);
    line.setAttribute('stroke', color);
    line.setAttribute('stroke-width', '1');
    line.setAttribute('stroke-opacity', '0.2');
    line.setAttribute('stroke-dasharray', '3 6');
    svg.appendChild(line);

    // Animated dot
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('r', '3');
    circle.setAttribute('fill', color);
    circle.setAttribute('opacity', '0.7');
    const anim = document.createElementNS('http://www.w3.org/2000/svg', 'animateMotion');
    anim.setAttribute('dur', '2.5s');
    anim.setAttribute('repeatCount', 'indefinite');
    anim.setAttribute('begin', `${delayMs}ms`);
    anim.setAttribute('path', `M${x1},${y1} L${x2},${y2}`);
    // Use path-based motion instead
    const motionPath = document.createElementNS('http://www.w3.org/2000/svg', 'animateMotion');
    motionPath.setAttribute('dur', '2.5s');
    motionPath.setAttribute('repeatCount', 'indefinite');
    motionPath.setAttribute('begin', `${delayMs}ms`);
    const pathD = `M0,0 L${x2-x1},${y2-y1}`;
    motionPath.setAttribute('path', pathD);
    circle.setAttribute('cx', x1);
    circle.setAttribute('cy', y1);
    circle.appendChild(motionPath);
    svg.appendChild(circle);
  }

  // Clear
  svg.innerHTML = '';

  // Layer labels
  function addLabel(x, y, text, color) {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', x); t.setAttribute('y', y);
    t.setAttribute('fill', color); t.setAttribute('font-family', 'Inter, sans-serif');
    t.setAttribute('font-size', '10'); t.setAttribute('font-weight', '700');
    t.setAttribute('letter-spacing', '2'); t.setAttribute('text-transform', 'uppercase');
    t.textContent = text;
    svg.appendChild(t);
  }

  addLabel(20, 50, 'USER INTERFACE', '#f97316');
  addLabel(20, 175, 'INTELLIGENCE LAYER', '#818cf8');
  addLabel(20, 310, 'DATA LAYER', '#64748b');

  // Top layer: Gradio (wide)
  isoBox(450, 85, 500, 60, 20, '#f97316', '#f97316', 'Gradio WebUI', ':7860');

  // Middle layer
  isoBox(250, 215, 200, 50, 18, '#22d3ee', '#22d3ee', 'Qdrant', ':6333');
  isoBox(450, 215, 200, 50, 18, '#f59e0b', '#f59e0b', 'Langfuse', ':3000');
  isoBox(650, 215, 200, 50, 18, '#10b981', '#10b981', 'Metaflow', ':3001');

  // Bottom layer
  isoBox(175, 350, 160, 44, 15, '#64748b', '#64748b', 'PostgreSQL', ':5432');
  isoBox(350, 350, 160, 44, 15, '#64748b', '#64748b', 'ClickHouse', ':8123');
  isoBox(525, 350, 160, 44, 15, '#64748b', '#64748b', 'Redis', ':6379');
  isoBox(700, 350, 160, 44, 15, '#64748b', '#64748b', 'MinIO', ':9090');

  // Flow connections: Gradio → Intelligence layer
  addFlowDots(350, 115, 250, 190, '#f97316', 0);
  addFlowDots(450, 115, 450, 190, '#f97316', 400);
  addFlowDots(550, 115, 650, 190, '#f97316', 800);

  // Intelligence → Data layer
  addFlowDots(250, 250, 175, 325, '#22d3ee', 200);
  addFlowDots(450, 250, 350, 325, '#f59e0b', 600);
  addFlowDots(650, 250, 525, 325, '#10b981', 1000);
  addFlowDots(650, 250, 700, 325, '#10b981', 1200);
}

const BOOT_SERVICES = [
  { name: 'PostgreSQL', port: 5432 },
  { name: 'ClickHouse', port: 8123 },
  { name: 'Redis', port: 6379 },
  { name: 'MinIO', port: 9090 },
  { name: 'Qdrant', port: 6333 },
  { name: 'Langfuse', port: 3000 },
  { name: 'Metaflow', port: 3001 },
  { name: 'Gradio WebUI', port: 7860 },
];

let bootAnimDone = false;
function runBootSequence() {
  if (bootAnimDone) return;
  bootAnimDone = true;
  const terminal = document.getElementById('boot-terminal');
  BOOT_SERVICES.forEach((svc, i) => {
    setTimeout(() => {
      const line = document.createElement('div');
      line.className = 'boot-line';
      line.innerHTML = `<span class="boot-check">✓</span> ${svc.name} <span style="color:var(--text-dim);">:${svc.port}</span>`;
      terminal.appendChild(line);
      // Trigger transition
      requestAnimationFrame(() => line.classList.add('visible'));
    }, 600 + i * 300);
  });
  // "All services up" after all
  setTimeout(() => {
    const line = document.createElement('div');
    line.className = 'boot-line';
    line.innerHTML = '<span style="color:var(--green);">All 8 services running.</span>';
    terminal.appendChild(line);
    requestAnimationFrame(() => line.classList.add('visible'));
  }, 600 + BOOT_SERVICES.length * 300 + 200);
}

// Build SVG immediately, animate on enter
buildIsometricSVG();
registerAnim(3,
  function enter() { runBootSequence(); },
  null
);
