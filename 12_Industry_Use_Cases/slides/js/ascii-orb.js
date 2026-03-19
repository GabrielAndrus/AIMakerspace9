/* ══════════════════════════════════════════════════════════════
   ASCII ORB — Amp-style character-density radial glow
   Renders on slides 1 (index 0) and 11 (index 10)
   ══════════════════════════════════════════════════════════════ */

(function() {
  // Character ramp: space → densest
  var CHARS = ' ..::-==++';
  var CELL  = 11;   // px per character cell
  var COLS  = 120;
  var ROWS  = 120;

  // Color stops: outer → inner (gruvbox-influenced warm palette)
  // Each: [r, g, b] — we'll interpolate and vary alpha by density
  var C_OUTER = [60, 90, 90];     // dark teal (matches bg)
  var C_MID   = [180, 120, 60];   // warm amber
  var C_INNER = [249, 115, 22];   // orange (--orange)

  var orbFrames = {};
  var orbRAFs   = {};

  function lerpColor(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t
    ];
  }

  function drawOrb(canvasId, time) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var ctx = canvas.getContext('2d');

    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    var cx = COLS / 2;
    var cy = ROWS / 2;
    var maxR = Math.sqrt(cx * cx + cy * cy);

    // Breathing: subtle radius pulse
    var breathe = 1.0 + 0.06 * Math.sin(time * 0.0008);

    ctx.textBaseline = 'top';

    for (var row = 0; row < ROWS; row++) {
      for (var col = 0; col < COLS; col++) {
        var dx = col - cx;
        var dy = (row - cy) * 1.1; // slight vertical stretch for oval
        var dist = Math.sqrt(dx * dx + dy * dy);

        // Normalize distance, apply breathing
        var nd = dist / (maxR * 0.55) / breathe;
        if (nd > 1.0) continue; // outside orb radius

        // Add noise ripple for organic feel
        var noise = 0.04 * Math.sin(dist * 0.8 + time * 0.002) +
                    0.03 * Math.sin(col * 0.5 + time * 0.001);
        nd = Math.max(0, Math.min(1, nd + noise));

        // Intensity: 1 at center, 0 at edge
        var intensity = 1.0 - nd;
        // Square for sharper falloff
        intensity = intensity * intensity;

        // Pick character from ramp
        var ci = Math.floor(intensity * (CHARS.length - 1));
        ci = Math.max(0, Math.min(CHARS.length - 1, ci));
        var ch = CHARS[ci];
        if (ch === ' ') continue;

        // Color interpolation: outer → mid → inner based on intensity
        var color;
        if (intensity < 0.5) {
          color = lerpColor(C_OUTER, C_MID, intensity * 2);
        } else {
          color = lerpColor(C_MID, C_INNER, (intensity - 0.5) * 2);
        }

        // Alpha based on intensity
        var alpha = 0.15 + intensity * 0.85;

        ctx.fillStyle = 'rgba(' +
          Math.round(color[0]) + ',' +
          Math.round(color[1]) + ',' +
          Math.round(color[2]) + ',' +
          alpha.toFixed(2) + ')';

        // Vary font size slightly for depth
        var fontSize = CELL * (0.85 + intensity * 0.25);
        ctx.font = fontSize + 'px "IBM Plex Mono", monospace';
        ctx.fillText(ch, col * CELL, row * CELL);
      }
    }
  }

  function animateOrb(canvasId) {
    function frame(t) {
      drawOrb(canvasId, t);
      orbRAFs[canvasId] = requestAnimationFrame(frame);
    }
    orbRAFs[canvasId] = requestAnimationFrame(frame);
  }

  function stopOrb(canvasId) {
    if (orbRAFs[canvasId]) {
      cancelAnimationFrame(orbRAFs[canvasId]);
      delete orbRAFs[canvasId];
    }
  }

  // Hook into slide lifecycle
  // Slide 1 (index 0)
  registerAnim(0,
    (function(origEnter) {
      return function(i) {
        animateOrb('ascii-orb-0');
        if (origEnter) origEnter(i);
      };
    })(slideAnims[0] && slideAnims[0].enter),
    (function(origLeave) {
      return function(i) {
        stopOrb('ascii-orb-0');
        if (origLeave) origLeave(i);
      };
    })(slideAnims[0] && slideAnims[0].leave)
  );

  // Slide 11 (index 10)
  registerAnim(10,
    (function(origEnter) {
      return function(i) {
        animateOrb('ascii-orb-10');
        if (origEnter) origEnter(i);
      };
    })(slideAnims[10] && slideAnims[10].enter),
    (function(origLeave) {
      return function(i) {
        stopOrb('ascii-orb-10');
        if (origLeave) origLeave(i);
      };
    })(slideAnims[10] && slideAnims[10].leave)
  );

  // Start immediately for slide 0 (it's active on load)
  animateOrb('ascii-orb-0');

  /* ════════════════════════════════════════════════════════════
     CORNER ASCII DECORATIONS — always visible, fixed position
     Both corners: Ambient Data Particle Field
     Bottom-left:  drifts upward-left from corner
     Top-right:    drifts downward-right from corner (mirrored)
     ════════════════════════════════════════════════════════════ */

  var CC = 11;  // cell size
  var CS = 50;  // grid dimension (50x50 = 550px)

  var DATA_CHARS = '0 1 . : + = 0 1 . :'.split(' ');

  // ── Create particle set for a corner ──
  function makeParticles(count) {
    var particles = [];
    for (var i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * CS,
        y: Math.random() * CS,
        ch: DATA_CHARS[Math.floor(Math.random() * DATA_CHARS.length)],
        vx: -0.008 - Math.random() * 0.012,
        vy: -0.015 - Math.random() * 0.02,
        spark: 0,
        sparkTime: 0
      });
    }
    return particles;
  }

  var blParticles = makeParticles(45);
  var trParticles = makeParticles(45);

  // ── Draw a particle field ──
  // originX/originY: the corner the particles emanate from (in grid coords)
  // driftX/driftY: direction multipliers for movement
  function drawParticleField(canvasId, particles, originX, originY, driftX, driftY, time) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.textBaseline = 'top';

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];

      // Move with slight sinusoidal sway
      p.x += p.vx * driftX + 0.005 * Math.sin(time * 0.0005 + i * 2) * driftX;
      p.y += p.vy * driftY;

      // Respawn when off visible area
      var offscreen = (driftY < 0) ? (p.y < -2 || p.x < -2) : (p.y > CS + 2 || p.x > CS + 2);
      if (offscreen) {
        if (driftY < 0) {
          // BL: respawn bottom-right area
          p.x = CS * 0.7 + Math.random() * CS * 0.4;
          p.y = CS * 0.7 + Math.random() * CS * 0.4;
        } else {
          // TR: respawn top-left area
          p.x = Math.random() * CS * 0.4;
          p.y = Math.random() * CS * 0.4;
        }
        p.ch = DATA_CHARS[Math.floor(Math.random() * DATA_CHARS.length)];
        p.spark = 0;
      }

      // Random spark event
      if (p.spark <= 0 && Math.random() < 0.0008) {
        p.spark = 1.0;
        p.sparkTime = time;
      }
      if (p.spark > 0) {
        p.spark = Math.max(0, 1.0 - (time - p.sparkTime) / 500);
      }

      // Fade toward origin corner
      var dist = Math.sqrt(Math.pow(p.x - originX, 2) + Math.pow(p.y - originY, 2));
      var fade = Math.max(0, 1.0 - dist / (CS * 1.3));
      if (fade <= 0) continue;

      var alpha = fade * (0.08 + p.spark * 0.3);
      var r = 201, g = 168, b = 76;
      if (p.spark > 0.5) { r = 226; g = 194; b = 112; }

      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha.toFixed(3) + ')';
      ctx.font = (CC * 0.8) + 'px "IBM Plex Mono", monospace';
      ctx.fillText(p.ch, p.x * CC, p.y * CC);
    }
  }

  // ── Glow layer for sparking particles ──
  function drawParticleGlow(glowCanvasId, particles, originX, originY, time) {
    var canvas = document.getElementById(glowCanvasId);
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      if (p.spark <= 0.05) continue;

      var dist = Math.sqrt(Math.pow(p.x - originX, 2) + Math.pow(p.y - originY, 2));
      var fade = Math.max(0, 1.0 - dist / (CS * 1.3));
      if (fade <= 0) continue;

      var gx = p.x * CC;
      var gy = p.y * CC;
      var glowR = 16 + p.spark * 10;
      var sparkGlow = ctx.createRadialGradient(gx, gy, 0, gx, gy, glowR);
      sparkGlow.addColorStop(0, 'rgba(201,168,76,' + (fade * p.spark * 0.2).toFixed(3) + ')');
      sparkGlow.addColorStop(1, 'rgba(201,168,76,0)');
      ctx.fillStyle = sparkGlow;
      ctx.fillRect(gx - glowR, gy - glowR, glowR * 2, glowR * 2);
    }
  }

  // Animate corners (always running, they're fixed/global)
  var cornerRAF;
  function animateCorners(t) {
    // Bottom-left: origin at bottom-left (CS, 0 in fade math), drifts up-left
    drawParticleField('ascii-corner-bl', blParticles, CS, 0, 1, -1, t);
    drawParticleGlow('glow-corner-bl', blParticles, CS, 0, t);
    // Top-right: origin at top-right (0, CS), drifts down-right
    drawParticleField('ascii-corner-tr', trParticles, 0, CS, -1, 1, t);
    drawParticleGlow('glow-corner-tr', trParticles, 0, CS, t);
    cornerRAF = requestAnimationFrame(animateCorners);
  }
  cornerRAF = requestAnimationFrame(animateCorners);
})();
