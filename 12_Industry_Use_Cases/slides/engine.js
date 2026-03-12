/* ══════════════════════════════════════════════════════════════
   GLOBAL INFRASTRUCTURE
   ══════════════════════════════════════════════════════════════ */

const GRADIO = 'http://localhost:7860';
const QDRANT = 'http://localhost:6333';
const LANGFUSE = 'http://localhost:3000';
const TOTAL = 10;
let cur = 0;

const slides = document.querySelectorAll('.slide');
const bar = document.getElementById('progress');
const num = document.getElementById('current');

/* ── Slide lifecycle system ── */
const slideAnims = {};
function registerAnim(i, enter, leave) {
  slideAnims[i] = { enter, leave };
}

function go(n) {
  if (n < 0 || n >= TOTAL || n === cur) return;
  const prev = cur;
  slides[prev].classList.remove('active');
  // fire leave
  if (slideAnims[prev] && slideAnims[prev].leave) {
    try { slideAnims[prev].leave(prev); } catch(e) { console.warn('leave error', e); }
  }
  cur = n;
  slides[cur].classList.add('active');
  num.textContent = cur + 1;
  bar.style.width = ((cur + 1) / TOTAL * 100) + '%';
  // fire enter
  if (slideAnims[cur] && slideAnims[cur].enter) {
    try { slideAnims[cur].enter(cur); } catch(e) { console.warn('enter error', e); }
  }
}

/* ── Keyboard + touch nav ── */
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); go(cur + 1); }
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); go(cur - 1); }
  else if (e.key === 'Home') { e.preventDefault(); go(0); }
  else if (e.key === 'End') { e.preventDefault(); go(TOTAL - 1); }
});
let tx = 0;
document.addEventListener('touchstart', e => { tx = e.touches[0].clientX; });
document.addEventListener('touchend', e => {
  const d = tx - e.changedTouches[0].clientX;
  if (Math.abs(d) > 60) d > 0 ? go(cur + 1) : go(cur - 1);
});

/* ── Gradio API helper with offline fallback ── */
async function gradioCall(fnName, args, fallback) {
  try {
    const r = await fetch(`${GRADIO}/api/${fnName}`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({data: args}), signal: AbortSignal.timeout(10000)
    });
    return (await r.json()).data;
  } catch { return fallback; }
}

/* ── Typewriter function ── */
function typewriter(el, text, speed = 45) {
  return new Promise(resolve => {
    el.innerHTML = '';
    let i = 0;
    const cursor = document.createElement('span');
    cursor.className = 'typewriter-cursor';
    function tick() {
      if (i < text.length) {
        // Handle HTML tags by inserting them all at once
        if (text[i] === '<') {
          const close = text.indexOf('>', i);
          if (close !== -1) {
            const tag = text.substring(i, close + 1);
            el.insertAdjacentHTML('beforeend', tag);
            i = close + 1;
          } else {
            el.insertAdjacentText('beforeend', text[i]);
            i++;
          }
        } else {
          el.insertAdjacentText('beforeend', text[i]);
          i++;
        }
        // Re-append cursor at end
        if (cursor.parentNode) cursor.remove();
        el.appendChild(cursor);
        setTimeout(tick, speed);
      } else {
        // Keep cursor blinking for 2s then remove
        setTimeout(() => { if (cursor.parentNode) cursor.remove(); }, 2000);
        resolve();
      }
    }
    tick();
  });
}
