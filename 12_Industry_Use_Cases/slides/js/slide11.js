/* ════════════════════════ SLIDE 11: Journey Recap & CTA (slide index 10) ════════════════════════ */

var journeyLineDrawn = false;

function drawJourneyLine() {
  var canvas = document.getElementById('journey-line-canvas');
  if (!canvas) return;

  var ctx = canvas.getContext('2d');
  var w = canvas.width;
  var h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  var nodes = document.querySelectorAll('.journey-node');
  if (nodes.length < 2) return;

  // Get node positions (proportional)
  var colors = [
    '#f97316', '#818cf8', '#22d3ee', '#f59e0b', '#a855f7', '#ef4444'
  ];

  var step = w / (nodes.length - 1);

  // Draw gradient line
  for (var i = 0; i < nodes.length - 1; i++) {
    var x1 = step * i;
    var x2 = step * (i + 1);
    var grad = ctx.createLinearGradient(x1, 0, x2, 0);
    grad.addColorStop(0, colors[i]);
    grad.addColorStop(1, colors[i + 1]);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, h / 2);
    ctx.lineTo(x2, h / 2);
    ctx.stroke();
  }
}

function animateJourneyLine(callback) {
  if (journeyLineDrawn) { if (callback) callback(); return; }
  journeyLineDrawn = true;

  var canvas = document.getElementById('journey-line-canvas');
  if (!canvas) { if (callback) callback(); return; }

  var ctx = canvas.getContext('2d');
  var w = canvas.width;
  var h = canvas.height;

  var colors = [
    '#f97316', '#818cf8', '#22d3ee', '#f59e0b', '#a855f7', '#ef4444'
  ];

  var nodes = document.querySelectorAll('.journey-node');
  var step = w / (nodes.length - 1);
  var totalLength = w;
  var progress = 0;
  var speed = totalLength / 40; // ~40 frames

  function frame() {
    progress += speed;
    if (progress > totalLength) progress = totalLength;

    ctx.clearRect(0, 0, w, h);

    for (var i = 0; i < nodes.length - 1; i++) {
      var x1 = step * i;
      var x2 = step * (i + 1);

      // Clip to progress
      if (x1 >= progress) break;
      var drawTo = Math.min(x2, progress);

      var grad = ctx.createLinearGradient(x1, 0, drawTo, 0);
      grad.addColorStop(0, colors[i]);
      grad.addColorStop(1, colors[Math.min(i + 1, colors.length - 1)]);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, h / 2);
      ctx.lineTo(drawTo, h / 2);
      ctx.stroke();
    }

    if (progress < totalLength) {
      requestAnimationFrame(frame);
    } else {
      if (callback) callback();
    }
  }

  requestAnimationFrame(frame);
}

function animateJourneyNodes() {
  var nodes = document.querySelectorAll('.journey-node');
  nodes.forEach(function(node, i) {
    setTimeout(function() { node.classList.add('visible'); }, i * 200);
  });
}

function animateValueCards() {
  var cards = document.querySelectorAll('.value-card');
  cards.forEach(function(card, i) {
    setTimeout(function() { card.classList.add('visible'); }, 800 + i * 250);
  });
}

registerAnim(10,
  function enter() {
    animateJourneyLine(function() {
      animateJourneyNodes();
    });
    animateValueCards();
  },
  null
);
