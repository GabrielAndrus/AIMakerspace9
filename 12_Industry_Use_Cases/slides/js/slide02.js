/* ══════════════════════════════════════════════════════════════
   SLIDE 2: GAUNTLET ANIMATION (slide index 1)
   ══════════════════════════════════════════════════════════════ */

function buildGauntlet() {
  const svg = document.getElementById('gauntlet-svg');
  if (svg.childNodes.length > 0) return;

  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  // Main path
  const mainPath = el('path', {
    d: 'M40,130 L200,130 L230,80 L290,130 L350,130 L380,80 L420,180 L460,130 L560,130 L590,80 L640,180 L680,130 L780,130 L840,130 L920,130',
    stroke: '#2e3648', 'stroke-width': '3', fill: 'none', 'stroke-linecap': 'round'
  });
  svg.appendChild(mainPath);

  // Start label
  const startText = el('text', { x: '20', y: '125', fill: '#94a3b8', 'font-size': '11', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
  startText.textContent = 'Your Data';
  svg.appendChild(startText);

  // End label
  const endText = el('text', { x: '860', y: '125', fill: '#10b981', 'font-size': '11', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
  endText.textContent = 'Trained Model';
  svg.appendChild(endText);

  // Obstacle 1: Model selection fork
  const obs1 = el('g', { class: 'gauntlet-obstacle', id: 'obs-1' });
  const obs1Title = el('text', { x: '280', y: '30', 'text-anchor': 'middle', fill: '#e2e8f0', 'font-size': '13', 'font-weight': '800', 'font-family': 'Inter, sans-serif' });
  obs1Title.textContent = 'Which algorithm?';
  obs1.appendChild(obs1Title);
  const obs1Sub = el('text', { x: '280', y: '240', 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '10', 'font-family': 'Inter, sans-serif' });
  obs1Sub.textContent = 'Dozens of options, no guidance';
  obs1.appendChild(obs1Sub);
  const fork1bg = el('rect', { x: '200', y: '40', width: '160', height: '100', rx: '8', fill: 'rgba(239,68,68,0.06)', stroke: '#ef4444', 'stroke-width': '1', 'stroke-dasharray': '4 3' });
  obs1.appendChild(fork1bg);
  ['RF?', 'XGB?', 'SVM?', 'LGBM?'].forEach((t, i) => {
    const txt = el('text', { x: String(220 + (i % 2) * 80), y: String(68 + Math.floor(i / 2) * 30), fill: '#ef4444', 'font-size': '11', 'font-weight': '600', 'font-family': 'JetBrains Mono, monospace' });
    txt.textContent = t;
    obs1.appendChild(txt);
  });
  const x1 = el('text', { x: '270', y: '160', fill: '#ef4444', 'font-size': '18', 'font-weight': '900', 'text-anchor': 'middle' });
  x1.textContent = '\u2716';
  obs1.appendChild(x1);
  svg.appendChild(obs1);

  // Obstacle 2: Method confusion
  const obs2 = el('g', { class: 'gauntlet-obstacle', id: 'obs-2' });
  const obs2Title = el('text', { x: '470', y: '30', 'text-anchor': 'middle', fill: '#e2e8f0', 'font-size': '13', 'font-weight': '800', 'font-family': 'Inter, sans-serif' });
  obs2Title.textContent = 'Which training method?';
  obs2.appendChild(obs2Title);
  const obs2Sub = el('text', { x: '470', y: '240', 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '10', 'font-family': 'Inter, sans-serif' });
  obs2Sub.textContent = 'Each needs different data formats';
  obs2.appendChild(obs2Sub);
  const fork2bg = el('rect', { x: '400', y: '40', width: '140', height: '100', rx: '8', fill: 'rgba(239,68,68,0.06)', stroke: '#ef4444', 'stroke-width': '1', 'stroke-dasharray': '4 3' });
  obs2.appendChild(fork2bg);
  ['SFT?', 'DPO?', 'GRPO?'].forEach((t, i) => {
    const txt = el('text', { x: String(430 + i * 40), y: '80', fill: '#ef4444', 'font-size': '11', 'font-weight': '600', 'font-family': 'JetBrains Mono, monospace' });
    txt.textContent = t;
    obs2.appendChild(txt);
  });
  const x2 = el('text', { x: '470', y: '160', fill: '#ef4444', 'font-size': '18', 'font-weight': '900', 'text-anchor': 'middle' });
  x2.textContent = '\u2716';
  obs2.appendChild(x2);
  svg.appendChild(obs2);

  // Obstacle 3: Error wall
  const obs3 = el('g', { class: 'gauntlet-obstacle', id: 'obs-3' });
  const obs3Title = el('text', { x: '685', y: '30', 'text-anchor': 'middle', fill: '#e2e8f0', 'font-size': '13', 'font-weight': '800', 'font-family': 'Inter, sans-serif' });
  obs3Title.textContent = 'Cryptic error messages';
  obs3.appendChild(obs3Title);
  const obs3Sub = el('text', { x: '685', y: '240', 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '10', 'font-family': 'Inter, sans-serif' });
  obs3Sub.textContent = 'Hours of debugging Stack Overflow';
  obs3.appendChild(obs3Sub);
  const fork3bg = el('rect', { x: '600', y: '40', width: '170', height: '100', rx: '8', fill: 'rgba(239,68,68,0.06)', stroke: '#ef4444', 'stroke-width': '1', 'stroke-dasharray': '4 3' });
  obs3.appendChild(fork3bg);
  ['CUDA OOM', 'NaN loss', 'KeyError', 'shapes \u2260'].forEach((t, i) => {
    const txt = el('text', { x: String(620 + (i % 2) * 80), y: String(68 + Math.floor(i / 2) * 28), fill: '#ef4444', 'font-size': '10', 'font-weight': '500', 'font-family': 'JetBrains Mono, monospace' });
    txt.textContent = t;
    obs3.appendChild(txt);
  });
  const x3 = el('text', { x: '685', y: '160', fill: '#ef4444', 'font-size': '18', 'font-weight': '900', 'text-anchor': 'middle' });
  x3.textContent = '\u2716';
  obs3.appendChild(x3);
  svg.appendChild(obs3);

  // Animated traveler dot
  const dot = el('circle', { r: '6', fill: '#f97316', id: 'gauntlet-dot' });
  const motion = el('animateMotion', {
    dur: '4s', fill: 'freeze', begin: 'indefinite',
    path: 'M40,130 L200,130'
  });
  dot.appendChild(motion);
  svg.appendChild(dot);
}

let gauntletPlayed = false;
function playGauntlet() {
  if (gauntletPlayed) return;
  gauntletPlayed = true;
  ['obs-1', 'obs-2', 'obs-3'].forEach((id, i) => {
    setTimeout(() => {
      document.getElementById(id)?.classList.add('visible');
    }, 600 + i * 800);
  });
}

registerAnim(1,
  function enter() { buildGauntlet(); playGauntlet(); },
  null
);
