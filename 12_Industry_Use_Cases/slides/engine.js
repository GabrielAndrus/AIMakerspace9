/* ══════════════════════════════════════════════════════════════
   GLOBAL INFRASTRUCTURE
   ══════════════════════════════════════════════════════════════ */

const GRADIO = 'http://192.168.1.33:7860';
const QDRANT = 'http://192.168.1.33:6333';
const LANGFUSE = 'http://192.168.1.33:3000';
const TOTAL = 11;
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

/* ── Gradio API helper (Gradio 6.x two-step call pattern) ── */
async function gradioCall(fnName, args, timeoutMs) {
  const timeout = timeoutMs || 120000;
  // Step 1: POST to get event_id
  const postResp = await fetch(`${GRADIO}/gradio_api/call/${fnName}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: args }),
    signal: AbortSignal.timeout(15000)
  });
  if (!postResp.ok) {
    const text = await postResp.text().catch(() => '');
    throw new Error('Gradio POST failed: ' + postResp.status + (text ? ' - ' + text.slice(0,200) : ''));
  }
  const postId = await postResp.json();
  const eventId = postId.event_id;
  if (!eventId) throw new Error('No event_id from Gradio');
  
  // Step 2: GET results via SSE endpoint
  const getResp = await fetch(`${GRADIO}/gradio_api/call/${fnName}/${eventId}`, {
    signal: AbortSignal.timeout(timeout)
  });
  if (!getResp.ok) {
    const text = await getResp.text().catch(() => '');
    throw new Error('Gradio GET failed: ' + getResp.status + (text ? ' - ' + text.slice(0,200) : ''));
  }
  
  // Parse SSE stream - look for "data: [...]" lines
  const text = await getResp.text();
  console.log('[gradioCall] Raw response for', fnName, ':', text.slice(0, 500));
  const lines = text.split('\n');
  let resultData = null;
  
  // Look for the complete event's data line (may have multiple events: pending, complete)
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.startsWith('data: ')) {
      try {
        let jsonStr = line.slice(6)
          .replace(/\bNaN\b/g, 'null')
          .replace(/\b-?Infinity\b/g, '"Infinity"');
        const parsed = JSON.parse(jsonStr);
        if (Array.isArray(parsed)) { 
          resultData = parsed; 
          break; 
        }
      } catch(e) {
        console.warn('[gradioCall] Failed to parse data line:', line.slice(0, 100), e);
      }
    }
  }
  
  // If no array found, try concatenating multi-line JSON (rare but possible)
  if (!resultData) {
    let dataBuffer = '';
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('data: ')) {
        dataBuffer += line.slice(6);
      } else if (line === '' && dataBuffer) {
        try {
          let jsonStr = dataBuffer
            .replace(/\bNaN\b/g, 'null')
            .replace(/\b-?Infinity\b/g, '"Infinity"');
          const parsed = JSON.parse(jsonStr);
          if (Array.isArray(parsed)) {
            resultData = parsed;
            break;
          }
        } catch(e) {}
        dataBuffer = '';
      }
    }
  }
  
  if (!resultData) throw new Error('Could not parse Gradio response. Raw: ' + text.slice(0,300));
  
  // Unwrap Gradio update objects: {"value": X, "__type__": "update"} -> X
  return resultData.map(function(item) {
    if (item && typeof item === 'object' && '__type__' in item && 'value' in item) {
      return item.value;
    }
    return item;
  });
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

/* ── Shared helper: set button loading state ── */
function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = loading;
  if (loading) {
    btn.dataset.origText = btn.textContent;
    btn.textContent = 'Working…';
  } else {
    btn.textContent = btn.dataset.origText || btn.textContent;
  }
}

/* ── Shared helper: escape HTML ── */
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
