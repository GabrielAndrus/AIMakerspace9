/* ══════════════════════════════════════════════════════════════
   SLIDE 7: LIVE QDRANT VECTOR SEARCH DEMO (slide index 6)
   ══════════════════════════════════════════════════════════════ */

let qdrantCollections = null;
let automlPoints = [];
let errorPoints = [];

/* ── Fetch Qdrant collections on slide enter ── */
async function fetchQdrantCollections() {
  const statusDot = document.getElementById('qd-status-dot');
  const statusText = document.getElementById('qd-status-text');
  const container = document.getElementById('qdrant-collections');

  try {
    const resp = await fetch(QDRANT + '/collections', { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    qdrantCollections = data.result?.collections || [];

    statusDot?.classList.remove('down');
    statusDot?.classList.add('healthy');
    if (statusText) statusText.textContent = 'Connected to Qdrant';

    // Fetch details for each collection
    const collectionsHtml = await Promise.all(qdrantCollections.map(async function(col) {
      try {
        const detailResp = await fetch(QDRANT + '/collections/' + col.name, { signal: AbortSignal.timeout(3000) });
        if (!detailResp.ok) return '';
        const detail = await detailResp.json();
        const pointsCount = detail.result?.points_count || 0;
        const vectorSize = detail.result?.config?.params?.vectors?.size || 'N/A';
        
        return '<div class="qd-collection-card" style="background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:10px 14px; min-width:160px;">' +
          '<div style="font-size:0.9rem; font-weight:700; color:var(--cyan);">' + col.name + '</div>' +
          '<div style="font-size:0.72rem; color:var(--text-dim); margin-top:4px;">' +
            '<span style="color:var(--green);">' + pointsCount + '</span> vectors &middot; ' +
            '<span style="color:var(--amber);">' + vectorSize + 'd</span>' +
          '</div>' +
        '</div>';
      } catch (e) {
        return '';
      }
    }));

    if (container) container.innerHTML = collectionsHtml.join('');

    // Load sample points for browsing
    await loadSamplePoints();

  } catch (err) {
    console.error('[Qdrant] Error:', err);
    statusDot?.classList.remove('healthy');
    statusDot?.classList.add('down');
    if (statusText) statusText.textContent = 'Qdrant unavailable';
    if (container) container.innerHTML = '<span style="color:var(--red);">Could not connect to Qdrant</span>';
  }
}

/* ── Load sample points from collections for browsing ── */
async function loadSamplePoints() {
  try {
    // Fetch automl_knowledge points
    const autoResp = await fetch(QDRANT + '/collections/automl_knowledge/points/scroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit: 10, with_payload: true, with_vector: false }),
      signal: AbortSignal.timeout(5000)
    });
    if (autoResp.ok) {
      const autoData = await autoResp.json();
      automlPoints = autoData.result?.points || [];
    }

    // Fetch error_investigations points
    const errResp = await fetch(QDRANT + '/collections/error_investigations/points/scroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit: 10, with_payload: true, with_vector: false }),
      signal: AbortSignal.timeout(5000)
    });
    if (errResp.ok) {
      const errData = await errResp.json();
      errorPoints = errData.result?.points || [];
    }

    // Populate browse stage
    renderBrowseView();

  } catch (err) {
    console.error('[Qdrant] Load points error:', err);
  }
}

/* ── Render browse view with loaded points ── */
function renderBrowseView() {
  const container = document.getElementById('qdrant-browse');
  if (!container) return;

  let html = '';

  // Automl knowledge
  if (automlPoints.length > 0) {
    html += '<div style="margin-bottom:10px;">';
    html += '<div style="font-size:0.75rem; font-weight:700; color:var(--cyan); margin-bottom:6px;">automl_knowledge (' + automlPoints.length + ' shown)</div>';
    automlPoints.slice(0, 5).forEach(function(pt) {
      const title = pt.payload?.title || 'Untitled';
      const source = pt.payload?.source || '';
      const content = (pt.payload?.content || '').slice(0, 80);
      html += '<div style="background:rgba(34,211,238,0.05); border-left:2px solid var(--cyan); padding:6px 8px; margin-bottom:4px; font-size:0.68rem;">';
      html += '<strong style="color:var(--white);">' + escapeHtml(title) + '</strong>';
      if (source) html += ' <span style="color:var(--text-dim);">(' + escapeHtml(source) + ')</span>';
      html += '<br><span style="color:var(--text-sec);">' + escapeHtml(content) + '\u2026</span>';
      html += '</div>';
    });
    html += '</div>';
  }

  // Error investigations
  if (errorPoints.length > 0) {
    html += '<div style="margin-bottom:10px;">';
    html += '<div style="font-size:0.75rem; font-weight:700; color:var(--amber); margin-bottom:6px;">error_investigations (' + errorPoints.length + ' shown)</div>';
    errorPoints.slice(0, 3).forEach(function(pt) {
      const errType = pt.payload?.error_type || 'Error';
      const errMsg = (pt.payload?.error_message || '').slice(0, 60);
      html += '<div style="background:rgba(245,158,11,0.05); border-left:2px solid var(--amber); padding:6px 8px; margin-bottom:4px; font-size:0.68rem;">';
      html += '<strong style="color:var(--red);">' + escapeHtml(errType) + '</strong>: ';
      html += '<span style="color:var(--text-sec);">' + escapeHtml(errMsg) + '\u2026</span>';
      html += '</div>';
    });
    html += '</div>';
  }

  if (!html) {
    html = '<span style="color:var(--text-dim); font-size:0.72rem;">No documents loaded. Check Qdrant connection.</span>';
  }

  container.innerHTML = html;
}

/* ── Vector scatter plot ── */
function drawVectorScatter(results, query) {
  var canvas = document.getElementById('vector-scatter-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  var pad = 30;

  // Grid
  ctx.strokeStyle = 'rgba(235,219,178,0.06)';
  ctx.lineWidth = 1;
  for (var g = 0; g <= 4; g++) {
    var gx = pad + g * ((w - pad * 2) / 4);
    var gy = pad + g * ((h - pad * 2) / 4);
    ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, h - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w - pad, gy); ctx.stroke();
  }

  // Axis labels
  ctx.font = '8px "IBM Plex Mono", monospace';
  ctx.fillStyle = 'rgba(235,219,178,0.3)';
  ctx.textAlign = 'center';
  ctx.fillText('dim 1', w / 2, h - 8);
  ctx.save();
  ctx.translate(10, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('dim 2', 0, 0);
  ctx.restore();

  if (!results || results.length === 0) {
    ctx.fillStyle = 'rgba(235,219,178,0.15)';
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Search to visualize', w / 2, h / 2);
    return;
  }

  // Generate pseudo-2D positions from scores
  var qx = w / 2, qy = h / 2;

  // Query point (gold, center)
  ctx.beginPath();
  ctx.arc(qx, qy, 6, 0, Math.PI * 2);
  ctx.fillStyle = '#c9a84c';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(qx, qy, 12, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(201,168,76,0.15)';
  ctx.fill();

  // Label
  ctx.font = 'bold 8px "IBM Plex Mono", monospace';
  ctx.fillStyle = '#c9a84c';
  ctx.textAlign = 'center';
  ctx.fillText('query', qx, qy - 16);

  // Result points - spread around query based on score (closer = more similar)
  results.forEach(function(r, i) {
    var angle = (i / results.length) * Math.PI * 2 - Math.PI / 2;
    var dist = (1 - r.score) * (Math.min(w, h) / 2 - pad - 20) + 20;
    var rx = qx + Math.cos(angle) * dist;
    var ry = qy + Math.sin(angle) * dist;

    var isAutoml = r.collection === 'automl_knowledge';
    var color = isAutoml ? '#22d3ee' : '#f59e0b';

    // Connection line
    ctx.strokeStyle = color;
    ctx.globalAlpha = r.score * 0.4;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(qx, qy);
    ctx.lineTo(rx, ry);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Point
    ctx.beginPath();
    ctx.arc(rx, ry, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.6 + r.score * 0.4;
    ctx.fill();
    ctx.globalAlpha = 1;

    // Score label
    ctx.font = '7px "IBM Plex Mono", monospace';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.fillText((r.score * 100).toFixed(0) + '%', rx, ry - 8);
  });
}

/* ── Animated search flow ── */
function animateSearchFlow(callback) {
  var outEl = document.getElementById('qd-out-1');
  if (!outEl) { callback(); return; }

  outEl.style.display = 'block';
  outEl.innerHTML = '<span style="color:var(--cyan);">Encoding query...</span>';

  setTimeout(function() {
    outEl.innerHTML = '<span style="color:var(--amber);">Searching vector space...</span>';
    setTimeout(function() {
      callback();
    }, 400);
  }, 400);
}

/* ── Search Qdrant (simulated semantic search since we don't have embedding model) ── */
async function searchQdrant() {
  const queryInput = document.getElementById('qdrant-query');
  const query = (queryInput?.value || '').trim().toLowerCase();

  if (!query) {
    showQdrantOutput('<span style="color:var(--amber);">Enter a search query.</span>');
    return;
  }

  setLoading('btn-qd-search', true);

  // Animated search flow
  animateSearchFlow(function() {});

  // Simulate semantic search by matching keywords in payloads
  await new Promise(r => setTimeout(r, 600));

  try {
    const results = [];
    
    // Search automl_knowledge
    automlPoints.forEach(function(pt) {
      const title = (pt.payload?.title || '').toLowerCase();
      const content = (pt.payload?.content || '').toLowerCase();
      const source = (pt.payload?.source || '').toLowerCase();
      
      let score = 0;
      query.split(/\s+/).forEach(function(term) {
        if (term.length < 2) return;
        if (title.includes(term)) score += 0.3;
        if (content.includes(term)) score += 0.15;
        if (source.includes(term)) score += 0.1;
      });

      // Boost for exact phrase matches
      if (content.includes(query)) score += 0.25;

      if (score > 0) {
        results.push({
          collection: 'automl_knowledge',
          title: pt.payload?.title || 'Untitled',
          content: pt.payload?.content || '',
          source: pt.payload?.source || '',
          score: Math.min(0.99, score)
        });
      }
    });

    // Search error_investigations
    errorPoints.forEach(function(pt) {
      const errType = (pt.payload?.error_type || '').toLowerCase();
      const errMsg = (pt.payload?.error_message || '').toLowerCase();
      const rec = (pt.payload?.recommendation || '').toLowerCase();
      
      let score = 0;
      query.split(/\s+/).forEach(function(term) {
        if (term.length < 2) return;
        if (errType.includes(term)) score += 0.35;
        if (errMsg.includes(term)) score += 0.25;
        if (rec.includes(term)) score += 0.1;
      });

      if (errMsg.includes(query) || rec.includes(query)) score += 0.2;

      if (score > 0) {
        results.push({
          collection: 'error_investigations',
          title: pt.payload?.error_type || 'Error',
          content: pt.payload?.recommendation || '',
          source: errType + ': ' + (pt.payload?.error_message || '').slice(0, 50),
          score: Math.min(0.99, score)
        });
      }
    });

    // Sort by score
    results.sort(function(a, b) { return b.score - a.score; });
    const topResults = results.slice(0, 5);
    if (topResults.length === 0) {
      showQdrantOutput('<span style="color:var(--text-dim);">No matching documents found.</span>');
      document.getElementById('qdrant-results').innerHTML = '<span style="color:var(--text-dim); font-size:0.72rem;">Try queries like "SVM", "GRPO reward", "CUDA OOM", "feature mismatch"</span>';
      drawVectorScatter([], query);
    } else {
      showQdrantOutput('<span style="color:var(--green);">\u2713 Found ' + topResults.length + ' relevant documents</span>');
      document.getElementById('qd-stage-2')?.classList.add('visible');
      renderQdrantResults(topResults, query);
      drawVectorScatter(topResults, query);
    }

  } catch (err) {
    showQdrantOutput('<span style="color:var(--red);">\u2717 Search error: ' + err.message + '</span>');
  } finally {
    setLoading('btn-qd-search', false);
  }
}

/* ── Show output in stage 1 ── */
function showQdrantOutput(html) {
  const el = document.getElementById('qd-out-1');
  if (el) { el.style.display = 'block'; el.innerHTML = html; }
}

/* ── Render search results with highlighted matches ── */
function renderQdrantResults(results, query) {
  const container = document.getElementById('qdrant-results');
  if (!container) return;

  let html = '';

  results.forEach(function(r, i) {
    const scorePct = (r.score * 100).toFixed(0);
    const scoreColor = r.score > 0.7 ? 'var(--green)' : r.score > 0.4 ? 'var(--amber)' : 'var(--text-dim)';

    // Highlight matching terms in content
    const highlightedContent = highlightMatches(r.content, query);
    const truncatedContent = truncateText(highlightedContent, 200);

    html += '<div class="qd-result-card" style="background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:10px; margin-bottom:8px;">';

    // Header with score
    html += '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">';
    html += '<div style="font-size:0.75rem; font-weight:700; color:' + (r.collection === 'automl_knowledge' ? 'var(--cyan)' : 'var(--amber)') + ';">' + escapeHtml(r.title) + '</div>';
    html += '<div style="font-size:0.72rem; font-weight:600; color:' + scoreColor + ';">' + scorePct + '%</div>';
    html += '</div>';

    // Source
    if (r.source) {
      html += '<div style="font-size:0.65rem; color:var(--text-dim); margin-bottom:4px;">' + escapeHtml(r.source) + '</div>';
    }

    // Content snippet
    html += '<div style="font-size:0.7rem; color:var(--text-sec); line-height:1.4;">' + truncatedContent + '</div>';

    // Collection badge
    html += '<div style="margin-top:6px;"><span style="font-size:0.6rem; background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:3px; color:var(--text-dim);">' + r.collection + '</span></div>';

    html += '</div>';
  });

  container.innerHTML = html;
}

/* ── Highlight matching terms in text ── */
function highlightMatches(text, query) {
  if (!text || !query) return escapeHtml(text || '');
  
  const escapedText = escapeHtml(text);
  const terms = query.toLowerCase().split(/\s+/).filter(function(t) { return t.length >= 2; });
  
  let result = escapedText;
  terms.forEach(function(term) {
    // Case-insensitive highlight
    const regex = new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
    result = result.replace(regex, '<mark style="background:rgba(34,211,238,0.3); color:var(--cyan); padding:0 2px; border-radius:2px;">$1</mark>');
  });
  
  return result;
}

/* ── Truncate text to max length ── */
function truncateText(html, maxLen) {
  // Strip HTML tags for length counting
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  const text = tmp.textContent || '';
  
  if (text.length <= maxLen) return html;
  
  // Find a good break point
  let truncated = html.slice(0, maxLen + 50); // Extra for HTML tags
  
  // Try to end at a sentence or word boundary
  const lastPeriod = Math.max(truncated.lastIndexOf('.'), truncated.lastIndexOf('\u2026'));
  if (lastPeriod > maxLen * 0.5) {
    truncated = truncated.slice(0, lastPeriod + 1);
  }
  
  return truncated + '\u2026';
}

/* ── Escape HTML entities ── */
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/* ── Register slide lifecycle ── */
registerAnim(6,
  function enter() {
    fetchQdrantCollections();
    document.getElementById('qd-stage-1')?.classList.add('visible');
    drawVectorScatter([], '');
  },
  function leave() {
    // Cleanup if needed
  }
);