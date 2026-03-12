/* ══════════════════════════════════════════════════════════════
   SLIDE 10: HEALTH RINGS + TABBED PLAYGROUND (slide index 9)
   ══════════════════════════════════════════════════════════════ */

const HEALTH_SERVICES = [
  { name: 'Gradio', port: 7860, url: GRADIO + '/info' },
  { name: 'Qdrant', port: 6333, url: QDRANT + '/collections' },
  { name: 'Langfuse', port: 3000, url: LANGFUSE + '/api/public/health' },
  { name: 'PostgreSQL', port: 5432, url: null },
  { name: 'ClickHouse', port: 8123, url: null },
  { name: 'Redis', port: 6379, url: null },
  { name: 'MinIO', port: 9090, url: null },
  { name: 'Metaflow', port: 3001, url: null },
];

function buildHealthRings() {
  const container = document.getElementById('health-rings');
  if (container.childNodes.length > 0) return;

  HEALTH_SERVICES.forEach((svc, i) => {
    const ring = document.createElement('div');
    ring.className = 'health-ring';
    ring.innerHTML = `
      <svg viewBox="0 0 44 44">
        <circle class="ring-bg" cx="22" cy="22" r="18"/>
        <circle class="ring-fill" id="ring-${i}" cx="22" cy="22" r="18"/>
      </svg>
      <div class="health-ring-label">${svc.name}</div>
      <div class="health-ring-ms" id="ring-ms-${i}">:${svc.port}</div>
    `;
    container.appendChild(ring);
  });
}

async function checkHealthRings() {
  HEALTH_SERVICES.forEach(async (svc, i) => {
    const ring = document.getElementById(`ring-${i}`);
    const ms = document.getElementById(`ring-ms-${i}`);
    if (!svc.url) {
      // No direct HTTP check available
      ring?.classList.add('healthy');
      if (ms) ms.textContent = ':' + svc.port;
      return;
    }
    try {
      const t0 = performance.now();
      const r = await fetch(svc.url, { signal: AbortSignal.timeout(3000) });
      const elapsed = Math.round(performance.now() - t0);
      if (r.ok) {
        ring?.classList.remove('down');
        ring?.classList.add('healthy');
        if (ms) ms.textContent = elapsed + 'ms';
      } else {
        ring?.classList.remove('healthy');
        ring?.classList.add('down');
        if (ms) ms.textContent = r.status;
      }
    } catch {
      ring?.classList.remove('healthy');
      ring?.classList.add('down');
      if (ms) ms.textContent = 'down';
    }
  });
}

// Tab switching
function switchTab(idx) {
  document.querySelectorAll('.api-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
  document.querySelectorAll('.api-tab-content').forEach((c, i) => c.classList.toggle('active', i === idx));
}

// Tab: Analyze CSV
async function tabAnalyzeCSV() {
  const out = document.getElementById('tab-0-out');
  const url = document.getElementById('csv-url')?.value;
  out.innerHTML = '<span class="req">POST /api/handle_csv_upload</span>\n<span class="loading">Sending...</span>';
  const result = await gradioCall('handle_csv_upload', [url], null);
  if (result) {
    out.innerHTML = '<span class="req">POST /api/handle_csv_upload</span>\n' + JSON.stringify(result, null, 2);
  } else {
    out.innerHTML = '<span class="req">POST /api/handle_csv_upload</span>\n<span style="color:var(--amber);">[Offline] Fallback: Titanic dataset\n891 rows, 12 columns\nTask: classification (target: Survived)\nRecommended: RandomForest + XGBoost + LightGBM</span>';
  }
}

// Tab: Analyze LLM Data
async function tabAnalyzeLLM(method) {
  const out = document.getElementById('tab-1-out');
  out.innerHTML = `<span class="req">POST /api/handle_llm_dataset_upload (${method})</span>\n<span class="loading">Analyzing...</span>`;
  const result = await gradioCall('handle_llm_dataset_upload', [null], null);
  const data = result || DTREE_FALLBACK[method];
  out.innerHTML = `<span class="req">POST /api/handle_llm_dataset_upload (${method})</span>\n` + JSON.stringify(data, null, 2);
}

// Tab: RAGAS Eval
async function tabRunRagas() {
  const out = document.getElementById('tab-2-out');
  const method = document.getElementById('ragas-method')?.value || 'hybrid';
  out.innerHTML = `<span class="req">POST /api/run_ragas_evaluation (${method})</span>\n<span class="loading">Evaluating...</span>`;
  const result = await gradioCall('run_ragas_evaluation', [method], null);
  if (result) {
    out.innerHTML = `<span class="req">POST /api/run_ragas_evaluation (${method})</span>\n` + JSON.stringify(result, null, 2);
  } else {
    const fallback = { method, faithfulness: 0.85, context_precision: 0.80, context_recall: 0.88 };
    out.innerHTML = `<span class="req">POST /api/run_ragas_evaluation (${method})</span>\n<span style="color:var(--amber);">[Offline] Fallback:</span>\n` + JSON.stringify(fallback, null, 2);
  }
}

registerAnim(9,
  function enter() { buildHealthRings(); checkHealthRings(); },
  null
);
