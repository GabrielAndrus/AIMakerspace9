/* ══════════════════════════════════════════════════════════════
   SLIDE 3: OUR SOLUTION — Platform / Agents / Customers (slide index 2)
   ══════════════════════════════════════════════════════════════ */

/* ── DGX Spark ASCII box diagram ── */
function drawDGXSpark(canvas) {
  var ctx = canvas.getContext('2d');
  var w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.font = '13px "IBM Plex Mono", monospace';
  ctx.textBaseline = 'top';

  // Box-drawing chars via Unicode escapes to prevent encoding corruption
  var TL='\u250C',TR='\u2510',BL='\u2514',BR='\u2518',H='\u2500',V='\u2502';
  var DTL='\u2554',DTR='\u2557',DBL='\u255A',DBR='\u255D',DH='\u2550',DV='\u2551';
  var hr = H.repeat(31);
  var dhr = DH.repeat(19);
  var sh = H.repeat(5);

  var lines = [
    TL+hr+TR,
    V+'    '+DTL+dhr+DTR+'      '+V,
    V+'    '+DV+'   DGX Spark       '+DV+'      '+V,
    V+'    '+DV+'   128GB \u00B7 Grace   '+DV+'      '+V,
    V+'    '+DBL+dhr+DBR+'      '+V,
    V+'                               '+V,
    V+'  '+TL+sh+TR+' '+TL+sh+TR+' '+TL+sh+TR+'      '+V,
    V+'  '+V+' GPU '+V+' '+V+' GPU '+V+' '+V+' NVMe'+V+'      '+V,
    V+'  '+BL+sh+BR+' '+BL+sh+BR+' '+BL+sh+BR+'      '+V,
    V+'                               '+V,
    V+'  docker compose up -d  \u2713      '+V,
    BL+hr+BR
  ];

  var BOX_CHARS = TL+TR+BL+BR+V+H+DTL+DTR+DBL+DBR+DV+DH;

  var lineH = 17;
  var startY = (h - lines.length * lineH) / 2;

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var y = startY + i * lineH;

    // Color the box-drawing chars in dim green, text in brighter
    for (var j = 0; j < line.length; j++) {
      var ch = line[j];
      var x = (w - lines[0].length * 8) / 2 + j * 8;

      if (BOX_CHARS.indexOf(ch) >= 0) {
        ctx.fillStyle = 'rgba(16,185,129,0.4)';
      } else if (ch === '\u2713') {
        ctx.fillStyle = '#b8bb26';
      } else if (i >= 2 && i <= 3 && j > 8 && j < 28) {
        ctx.fillStyle = '#ebdbb2';
      } else {
        ctx.fillStyle = 'rgba(235,219,178,0.35)';
      }
      ctx.fillText(ch, x, y);
    }
  }
}

/* ── Agent Network Canvas ── */
var agentAnimFrame = null;

var AGENTS = [
  { label: 'Orchestrator',     color: '#ffffff',  x: 0.50, y: 0.22 },
  { label: 'Model Selection',  color: '#f97316',  x: 0.20, y: 0.45 },
  { label: 'Dataset Analyzer', color: '#818cf8',  x: 0.80, y: 0.45 },
  { label: 'Error Investigator', color: '#ef4444', x: 0.20, y: 0.72 },
  { label: 'Feature Engineer', color: '#22d3ee',  x: 0.50, y: 0.82 },
  { label: 'Training Monitor', color: '#f59e0b',  x: 0.80, y: 0.72 }
];

var AGENT_EDGES = [
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],  // orchestrator hub
  [1, 3], [2, 5], [1, 4], [2, 4]            // cross-connections
];

var agentPulses = [];

function drawAgentNetwork(canvas, t) {
  var ctx = canvas.getContext('2d');
  var w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  var pad = 30;
  var aw = w - pad * 2, ah = h - pad * 2;

  // Draw edges
  for (var e = 0; e < AGENT_EDGES.length; e++) {
    var from = AGENTS[AGENT_EDGES[e][0]];
    var to   = AGENTS[AGENT_EDGES[e][1]];
    var x1 = pad + from.x * aw, y1 = pad + from.y * ah;
    var x2 = pad + to.x * aw,   y2 = pad + to.y * ah;

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Spawn pulses
  if (Math.random() < 0.04) {
    var edgeIdx = Math.floor(Math.random() * AGENT_EDGES.length);
    agentPulses.push({ edge: edgeIdx, progress: 0 });
  }

  // Draw pulses
  for (var p = agentPulses.length - 1; p >= 0; p--) {
    var pulse = agentPulses[p];
    pulse.progress += 0.018;
    if (pulse.progress > 1) { agentPulses.splice(p, 1); continue; }

    var pe = AGENT_EDGES[pulse.edge];
    var pf = AGENTS[pe[0]], pt = AGENTS[pe[1]];
    var px = pad + (pf.x + (pt.x - pf.x) * pulse.progress) * aw;
    var py = pad + (pf.y + (pt.y - pf.y) * pulse.progress) * ah;
    var pc = pt.color;

    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = pc;
    ctx.globalAlpha = 0.7;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // Draw nodes
  for (var i = 0; i < AGENTS.length; i++) {
    var a = AGENTS[i];
    var nx = pad + a.x * aw;
    var ny = pad + a.y * ah;
    var isOrch = i === 0;
    var r = isOrch ? 20 : 14;

    // Glow ring on orchestrator
    if (isOrch) {
      var pulse2 = Math.sin(t * 0.002) * 0.5 + 0.5;
      ctx.beginPath();
      ctx.arc(nx, ny, r + 4 + pulse2 * 3, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,' + (0.06 + pulse2 * 0.06) + ')';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Node circle
    ctx.beginPath();
    ctx.arc(nx, ny, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fill();
    ctx.strokeStyle = a.color;
    ctx.lineWidth = isOrch ? 2 : 1.5;
    ctx.stroke();

    // Label
    ctx.font = (isOrch ? 'bold 10px' : '9px') + ' "IBM Plex Mono", monospace';
    ctx.fillStyle = a.color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(a.label, nx, ny + r + 12);
  }
}

function animateAgentNetwork() {
  var canvas = document.getElementById('agent-network-canvas');
  if (!canvas) return;
  drawAgentNetwork(canvas, performance.now());
  agentAnimFrame = requestAnimationFrame(animateAgentNetwork);
}

/* ── Service tags ── */
var SERVICES = [
  { name: 'Gradio',     port: ':7860' },
  { name: 'Qdrant',     port: ':6333' },
  { name: 'Langfuse',   port: ':3000' },
  { name: 'Metaflow',   port: ':3001' },
  { name: 'PostgreSQL',  port: ':5432' },
  { name: 'ClickHouse', port: ':8123' },
  { name: 'Redis',      port: ':6379' },
  { name: 'MinIO',      port: ':9090' }
];

function buildServiceTags() {
  var container = document.getElementById('solution-services');
  if (!container || container.children.length > 0) return;

  for (var i = 0; i < SERVICES.length; i++) {
    var tag = document.createElement('span');
    tag.className = 'solution-service-tag';
    tag.textContent = SERVICES[i].name + ' ' + SERVICES[i].port;
    container.appendChild(tag);
  }
}

function animateServiceTags() {
  var tags = document.querySelectorAll('.solution-service-tag');
  tags.forEach(function(tag, i) {
    setTimeout(function() { tag.classList.add('visible'); }, 200 + i * 80);
  });
}

function animateCustomerCards() {
  var cards = document.querySelectorAll('.customer-card');
  cards.forEach(function(card, i) {
    setTimeout(function() { card.classList.add('visible'); }, 600 + i * 300);
  });
}

/* ── Lifecycle ── */
registerAnim(2,
  function enter() {
    drawDGXSpark(document.getElementById('dgx-spark-canvas'));
    buildServiceTags();
    animateServiceTags();
    agentPulses = [];
    animateAgentNetwork();
  },
  function leave() {
    if (agentAnimFrame) {
      cancelAnimationFrame(agentAnimFrame);
      agentAnimFrame = null;
    }
  }
);
