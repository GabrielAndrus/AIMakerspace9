/* ══════════════════════════════════════════════════════════════
   SLIDE 4: ASCII ARCHITECTURE + BOOT SEQUENCE (slide index 3)
   ══════════════════════════════════════════════════════════════ */

var ARCH_SERVICES = [
  { name: 'Gradio',     port: ':7860', color: '#f97316', layer: 0, col: 1.5 },
  { name: 'Qdrant',     port: ':6333', color: '#22d3ee', layer: 1, col: 0 },
  { name: 'Langfuse',   port: ':3000', color: '#f59e0b', layer: 1, col: 1 },
  { name: 'Metaflow',   port: ':3001', color: '#10b981', layer: 1, col: 2 },
  // Bottom row intentionally uses the same col mapping but spread wider
  { name: 'PostgreSQL', port: ':5432', color: '#928374', layer: 2, col: 0 },
  { name: 'ClickHouse', port: ':8123', color: '#928374', layer: 2, col: 1 },
  { name: 'Redis',      port: ':6379', color: '#928374', layer: 2, col: 2 },
  { name: 'MinIO',      port: ':9090', color: '#928374', layer: 2, col: 3 }
];

var ARCH_CONNECTIONS = [
  // Gradio → middle
  { from: 0, to: 1 }, { from: 0, to: 2 }, { from: 0, to: 3 },
  // Middle → bottom
  { from: 1, to: 4 }, { from: 2, to: 5 }, { from: 3, to: 6 }, { from: 3, to: 7 }
];

var archPulses = [];
var archAnimFrame = null;
var archServiceLit = [];

function getServicePos(svc, w, h) {
  var padX = 100, padY = 70;
  var layerH = (h - padY * 2) / 2;
  var y = padY + svc.layer * layerH;

  if (svc.layer === 0) {
    return { x: w / 2, y: y };
  } else if (svc.layer === 1) {
    var colW1 = (w - padX * 2) / 2;
    return { x: padX + svc.col * colW1, y: y };
  } else {
    var colW2 = (w - padX * 2) / 3;
    return { x: padX + svc.col * colW2, y: y };
  }
}

function drawArchBox(ctx, x, y, label, port, color, lit) {
  var bw = 160, bh = 70;
  var lx = x - bw / 2, ly = y - bh / 2;

  // Box-drawing style with ASCII corners
  ctx.strokeStyle = lit ? color : 'rgba(235,219,178,0.15)';
  ctx.lineWidth = lit ? 2 : 1.5;
  ctx.strokeRect(lx, ly, bw, bh);

  if (lit) {
    var r = parseInt(color.slice(1,3), 16);
    var g = parseInt(color.slice(3,5), 16);
    var b = parseInt(color.slice(5,7), 16);
    ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.06)';
    ctx.fillRect(lx, ly, bw, bh);
  }

  // Label
  ctx.font = 'bold 18px "IBM Plex Mono", monospace';
  ctx.fillStyle = lit ? color : 'rgba(235,219,178,0.4)';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x, y - 8);

  // Port
  ctx.font = '14px "IBM Plex Mono", monospace';
  ctx.fillStyle = lit ? 'rgba(235,219,178,0.5)' : 'rgba(235,219,178,0.2)';
  ctx.fillText(port, x, y + 16);
}

function drawArchitecture(canvas, t) {
  var ctx = canvas.getContext('2d');
  var w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Layer labels
  ctx.font = 'bold 14px "IBM Plex Mono", monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = 'rgba(249,115,22,0.5)';
  ctx.fillText('USER INTERFACE', 16, 38);
  ctx.fillStyle = 'rgba(129,140,248,0.5)';
  ctx.fillText('AI DECISION LAYER', 16, 225);
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.fillText('STORAGE & LOGGING', 16, 420);

  // Connections
  for (var c = 0; c < ARCH_CONNECTIONS.length; c++) {
    var conn = ARCH_CONNECTIONS[c];
    var fromSvc = ARCH_SERVICES[conn.from];
    var toSvc = ARCH_SERVICES[conn.to];
    var p1 = getServicePos(fromSvc, w, h);
    var p2 = getServicePos(toSvc, w, h);

    ctx.strokeStyle = 'rgba(235,219,178,0.06)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 6]);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y + 35);
    ctx.lineTo(p2.x, p2.y - 35);
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrow
    var ax = p2.x, ay = p2.y - 38;
    ctx.fillStyle = 'rgba(235,219,178,0.1)';
    ctx.beginPath();
    ctx.moveTo(ax, ay + 6);
    ctx.lineTo(ax - 5, ay - 3);
    ctx.lineTo(ax + 5, ay - 3);
    ctx.fill();
  }

  // Spawn pulses
  if (Math.random() < 0.03) {
    var ci = Math.floor(Math.random() * ARCH_CONNECTIONS.length);
    archPulses.push({ conn: ci, progress: 0 });
  }

  // Draw pulses
  for (var p = archPulses.length - 1; p >= 0; p--) {
    var pulse = archPulses[p];
    pulse.progress += 0.015;
    if (pulse.progress > 1) { archPulses.splice(p, 1); continue; }

    var pc = ARCH_CONNECTIONS[pulse.conn];
    var pf = getServicePos(ARCH_SERVICES[pc.from], w, h);
    var pt = getServicePos(ARCH_SERVICES[pc.to], w, h);
    var px = pf.x + (pt.x - pf.x) * pulse.progress;
    var py = (pf.y + 35) + ((pt.y - 35) - (pf.y + 35)) * pulse.progress;

    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = ARCH_SERVICES[pc.to].color;
    ctx.globalAlpha = 0.6;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // Draw service boxes
  for (var i = 0; i < ARCH_SERVICES.length; i++) {
    var svc = ARCH_SERVICES[i];
    var pos = getServicePos(svc, w, h);
    var lit = archServiceLit[i] || false;
    drawArchBox(ctx, pos.x, pos.y, svc.name, svc.port, svc.color, lit);
  }
}

function animateArch() {
  var canvas = document.getElementById('arch-canvas');
  if (!canvas) return;
  drawArchitecture(canvas, performance.now());
  archAnimFrame = requestAnimationFrame(animateArch);
}

/* ── Boot sequence (synced with diagram) ── */
var BOOT_ORDER = [
  { name: 'PostgreSQL', port: 5432, archIdx: 4 },
  { name: 'ClickHouse', port: 8123, archIdx: 5 },
  { name: 'Redis',      port: 6379, archIdx: 6 },
  { name: 'MinIO',      port: 9090, archIdx: 7 },
  { name: 'Qdrant',     port: 6333, archIdx: 1 },
  { name: 'Langfuse',   port: 3000, archIdx: 2 },
  { name: 'Metaflow',   port: 3001, archIdx: 3 },
  { name: 'Gradio WebUI', port: 7860, archIdx: 0 }
];

var bootAnimDone = false;

function runBootSequence() {
  if (bootAnimDone) return;
  bootAnimDone = true;

  var terminal = document.getElementById('boot-terminal');
  if (!terminal) return;
  var body = terminal.querySelector('.boot-terminal-body');
  if (!body) return;

  archServiceLit = [];

  BOOT_ORDER.forEach(function(svc, i) {
    setTimeout(function() {
      // Terminal line
      var line = document.createElement('div');
      line.className = 'boot-line';
      line.innerHTML = '<span class="boot-check">\u2713</span> ' + svc.name + ' <span style="color:#928374;">:' + svc.port + '</span>';
      body.appendChild(line);
      requestAnimationFrame(function() { line.classList.add('visible'); });

      // Light up service in diagram
      archServiceLit[svc.archIdx] = true;
    }, 600 + i * 300);
  });

  // "All services up" message
  setTimeout(function() {
    var line = document.createElement('div');
    line.className = 'boot-line';
    line.innerHTML = '<span style="color:#b8bb26;">All 8 services running.</span>';
    body.appendChild(line);
    requestAnimationFrame(function() { line.classList.add('visible'); });
  }, 600 + BOOT_ORDER.length * 300 + 200);
}

registerAnim(3,
  function enter() {
    archPulses = [];
    archServiceLit = [];
    animateArch();
    runBootSequence();
  },
  function leave() {
    if (archAnimFrame) {
      cancelAnimationFrame(archAnimFrame);
      archAnimFrame = null;
    }
  }
);
