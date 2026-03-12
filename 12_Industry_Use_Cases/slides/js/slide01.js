/* ══════════════════════════════════════════════════════════════
   SLIDE 1: PARTICLE NETWORK CANVAS
   ══════════════════════════════════════════════════════════════ */
const particleCanvas = document.getElementById('particle-canvas');
const pCtx = particleCanvas.getContext('2d');
let particles = [];
let particleRAF = null;

const PARTICLE_COLORS = [
  '#f97316', // orange - Gradio
  '#10b981', // green - Metaflow
  '#818cf8', // indigo - LangGraph
  '#22d3ee', // cyan - Qdrant
  '#f59e0b', // amber - Langfuse
];

function initParticles() {
  const w = particleCanvas.width = particleCanvas.offsetWidth;
  const h = particleCanvas.height = particleCanvas.offsetHeight;
  particles = [];
  const count = 45;
  for (let i = 0; i < count; i++) {
    particles.push({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      r: 3 + Math.random() * 3,
      color: PARTICLE_COLORS[i % PARTICLE_COLORS.length],
      phase: Math.random() * Math.PI * 2,
    });
  }
}

function drawParticles(time) {
  const w = particleCanvas.width;
  const h = particleCanvas.height;
  pCtx.clearRect(0, 0, w, h);

  const t = time * 0.001;

  // Update + draw connections
  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    p.x += p.vx + Math.sin(t + p.phase) * 0.3;
    p.y += p.vy + Math.cos(t * 0.7 + p.phase) * 0.3;
    // Wrap
    if (p.x < -20) p.x = w + 20;
    if (p.x > w + 20) p.x = -20;
    if (p.y < -20) p.y = h + 20;
    if (p.y > h + 20) p.y = -20;

    // Connections
    for (let j = i + 1; j < particles.length; j++) {
      const q = particles[j];
      const dx = p.x - q.x, dy = p.y - q.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 160) {
        pCtx.beginPath();
        pCtx.moveTo(p.x, p.y);
        pCtx.lineTo(q.x, q.y);
        pCtx.strokeStyle = `rgba(148,163,184,${0.12 * (1 - dist / 160)})`;
        pCtx.lineWidth = 1;
        pCtx.stroke();
      }
    }
  }
  // Draw nodes
  for (const p of particles) {
    pCtx.beginPath();
    pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    pCtx.fillStyle = p.color;
    pCtx.globalAlpha = 0.7;
    pCtx.fill();
    // Glow
    pCtx.beginPath();
    pCtx.arc(p.x, p.y, p.r * 2.5, 0, Math.PI * 2);
    const grad = pCtx.createRadialGradient(p.x, p.y, p.r, p.x, p.y, p.r * 2.5);
    grad.addColorStop(0, p.color.replace(')', ',0.15)').replace('rgb', 'rgba'));
    grad.addColorStop(1, 'transparent');
    pCtx.fillStyle = grad;
    pCtx.fill();
    pCtx.globalAlpha = 1;
  }

  particleRAF = requestAnimationFrame(drawParticles);
}

function startParticles() {
  initParticles();
  if (!particleRAF) particleRAF = requestAnimationFrame(drawParticles);
}
function stopParticles() {
  if (particleRAF) { cancelAnimationFrame(particleRAF); particleRAF = null; }
}

// Slide 1 lifecycle
let slide1Entered = false;
registerAnim(0,
  function enter() {
    startParticles();
    if (!slide1Entered) {
      slide1Entered = true;
      setTimeout(() => {
        typewriter(
          document.getElementById('typewriter-target'),
          'What if training ML models required <strong style="color:#f8fafc;">zero machine learning expertise?</strong>',
          40
        );
      }, 400);
    }
  },
  function leave() { stopParticles(); }
);

// Start particles immediately since slide 1 is active on load
window.addEventListener('resize', () => { if (cur === 0) initParticles(); });
