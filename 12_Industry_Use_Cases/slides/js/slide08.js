/* ══════════════════════════════════════════════════════════════
   SLIDE 8: LIVE LANGFUSE OBSERVABILITY DEMO (slide index 7)
   ══════════════════════════════════════════════════════════════ */

let langfuseTraces = [];
let selectedTraceId = null;

const SIMULATED_TRACES = [
  {
    id: 'trace-001',
    name: 'Model Selection Agent',
    type: 'agent',
    timestamp: '2 min ago',
    latency_ms: 3240,
    total_tokens: 1847,
    prompt_tokens: 412,
    completion_tokens: 1435,
    score: 0.92,
    score_label: 'accuracy',
    session_id: 'sess-auto-2024',
    metadata: { dataset: 'titanic.csv', target: 'Survived' },
    steps: [
      { name: 'Data Analysis', latency_ms: 450, tokens: 120 },
      { name: 'Feature Detection', latency_ms: 380, tokens: 95 },
      { name: 'Model Recommendation', latency_ms: 890, tokens: 680 },
      { name: 'Hyperparameter Selection', latency_ms: 1520, tokens: 952 }
    ],
    prompt: 'Analyze this dataset and recommend the best ML models for classification:\n\nDataset: titanic.csv\nRows: 891\nFeatures: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked\nTarget: Survived (binary)\nMissing values: Age (177), Cabin (687), Embarked (2)',
    generation: 'Based on the dataset characteristics:\n\n1. **RandomForestClassifier** - Best for mixed feature types, handles missing values well\n2. **GradientBoostingClassifier** - Strong performance on tabular data\n3. **LogisticRegression** - Good baseline, interpretable\n\nRecommended pipeline:\n- Impute Age with median\n- One-hot encode Sex and Embarked\n- Feature engineering: FamilySize = SibSp + Parch',
    observations: [
      { name: 'data_shape', value: '891 rows x 7 features' },
      { name: 'missing_pct', value: '19.9% Age, 77% Cabin' }
    ]
  },
  {
    id: 'trace-002',
    name: 'RAG Retrieval Query',
    type: 'rag',
    timestamp: '5 min ago',
    latency_ms: 890,
    total_tokens: 623,
    prompt_tokens: 45,
    completion_tokens: 578,
    score: 0.85,
    score_label: 'relevance',
    session_id: 'sess-rag-2024',
    metadata: { collection: 'automl_knowledge', k: 5 },
    steps: [
      { name: 'Embedding Query', latency_ms: 120, tokens: 45 },
      { name: 'Vector Search', latency_ms: 85, tokens: 0 },
      { name: 'Context Assembly', latency_ms: 200, tokens: 350 },
      { name: 'LLM Generation', latency_ms: 485, tokens: 228 }
    ],
    prompt: 'How do I fix CUDA out of memory errors during training?',
    generation: 'CUDA OOM errors occur when GPU memory is exhausted. Solutions:\n\n1. **Reduce batch size** - Start with 16, decrease if needed\n2. **Enable gradient checkpointing** - Trades compute for memory\n3. **Use mixed precision (fp16/bf16)** - Halves activation memory\n4. **Clear cache between runs** - `torch.cuda.empty_cache()`\n5. **Gradient accumulation** - Effective batch without memory cost',
    observations: [
      { name: 'retrieved_docs', value: '5' },
      { name: 'avg_similarity', value: '0.78' }
    ]
  },
  {
    id: 'trace-003',
    name: 'Training Step Monitor',
    type: 'training',
    timestamp: '12 min ago',
    latency_ms: 4500,
    total_tokens: 234,
    prompt_tokens: 0,
    completion_tokens: 0,
    score: 0.88,
    score_label: 'convergence',
    session_id: 'sess-train-2024',
    metadata: { model: 'RandomForest', n_estimators: 100 },
    steps: [
      { name: 'Data Split (80/20)', latency_ms: 50, tokens: 0 },
      { name: 'Cross-validation (5-fold)', latency_ms: 1200, tokens: 0 },
      { name: 'Model Training', latency_ms: 2800, tokens: 0 },
      { name: 'Evaluation Metrics', latency_ms: 450, tokens: 0 }
    ],
    prompt: '',
    generation: '',
    observations: [
      { name: 'train_accuracy', value: '94.2%' },
      { name: 'val_accuracy', value: '87.3%' },
      { name: 'f1_score', value: '0.86' },
      { name: 'best_params', value: 'n_estimators=100, max_depth=10' }
    ]
  },
  {
    id: 'trace-004',
    name: 'Error Investigation Agent',
    type: 'agent',
    timestamp: '18 min ago',
    latency_ms: 2100,
    total_tokens: 892,
    prompt_tokens: 156,
    completion_tokens: 736,
    score: 0.78,
    score_label: 'helpfulness',
    session_id: 'sess-error-2024',
    metadata: { error_type: 'ValueError', component: 'preprocessing' },
    steps: [
      { name: 'Error Classification', latency_ms: 180, tokens: 80 },
      { name: 'Root Cause Analysis', latency_ms: 450, tokens: 200 },
      { name: 'Solution Generation', latency_ms: 1470, tokens: 456 }
    ],
    prompt: 'Error during preprocessing:\n```\nValueError: Found input variables with inconsistent numbers of samples: [891, 712]\n```',
    generation: '**Root Cause**: Train/test split mismatch due to NaN handling.\n\n**Solution**:\nThe issue occurs because some rows have NaN in features but not target. Fix:\n\n```python\n# Drop rows with ANY missing values before split\ndf_clean = df.dropna(subset=feature_cols + [target])\nX_train, X_test, y_train, y_test = train_test_split(\n    df_clean[feature_cols], \n    df_clean[target],\n    test_size=0.2\n)\n```\n\nOr use an imputer pipeline to handle NaN values automatically.',
    observations: [
      { name: 'error_category', value: 'data_mismatch' },
      { name: 'confidence', value: 'high' }
    ]
  },
  {
    id: 'trace-005',
    name: 'LLM Fine-tuning Step',
    type: 'llm_training',
    timestamp: '25 min ago',
    latency_ms: 125000,
    total_tokens: 15240,
    prompt_tokens: 8400,
    completion_tokens: 6840,
    score: 0.95,
    score_label: 'loss_decrease',
    session_id: 'sess-ft-2024',
    metadata: { model: 'Qwen2.5-0.5B', method: 'SFT' },
    steps: [
      { name: 'Data Preparation', latency_ms: 5000, tokens: 0 },
      { name: 'Tokenization', latency_ms: 2000, tokens: 8400 },
      { name: 'Training Loop (3 epochs)', latency_ms: 115000, tokens: 6840 },
      { name: 'Checkpoint Save', latency_ms: 3000, tokens: 0 }
    ],
    prompt: '',
    generation: '',
    observations: [
      { name: 'initial_loss', value: '2.34' },
      { name: 'final_loss', value: '0.87' },
      { name: 'learning_rate', value: '1e-5' },
      { name: 'effective_batch', value: '32 (4 x 8 acc)' }
    ]
  }
];

async function checkLangfuseHealth() {
  const statusDot = document.getElementById('lf-status-dot');
  const statusText = document.getElementById('lf-status-text');

  try {
    const resp = await fetch(LANGFUSE + '/api/public/health', { signal: AbortSignal.timeout(5000) });
    if (resp.ok) {
      const data = await resp.json();
      statusDot?.classList.remove('down');
      statusDot?.classList.add('healthy');
      if (statusText) statusText.textContent = 'Connected to Langfuse v' + (data.version || '?');
      return true;
    }
  } catch (err) {
    console.log('[Langfuse] Health check failed:', err);
  }

  statusDot?.classList.remove('healthy');
  statusDot?.classList.add('down');
  if (statusText) statusText.textContent = 'Using simulated trace data';
  return false;
}

function renderTraceList() {
  const container = document.getElementById('langfuse-traces');
  if (!container) return;

  let html = '';
  
  SIMULATED_TRACES.forEach(function(trace, idx) {
    const typeIcon = trace.type === 'agent' ? '\u{1F916}' : 
                     trace.type === 'rag' ? '\u{1F50D}' :
                     trace.type === 'training' ? '\u2699\uFE0F' :
                     trace.type === 'llm_training' ? '\u{1F4DA}' : '\u{1F4CB}';
    
    const latencyColor = trace.latency_ms < 1000 ? 'var(--green)' : 
                         trace.latency_ms < 5000 ? 'var(--amber)' : 'var(--red)';
    
    const scoreColor = trace.score >= 0.85 ? 'var(--green)' :
                       trace.score >= 0.70 ? 'var(--amber)' : 'var(--red)';

    const latencyStr = trace.latency_ms >= 1000 ? 
                       (trace.latency_ms / 1000).toFixed(1) + 's' : 
                       trace.latency_ms + 'ms';

    html += '<div class="lf-trace-card" data-trace-id="' + trace.id + '" onclick="selectTrace(\'' + trace.id + '\')" style="background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:10px 12px; margin-bottom:6px; cursor:pointer; transition:border-color 0.15s;">';
    
    html += '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">';
    html += '<div style="font-size:0.8rem; font-weight:600; color:var(--white);">' + typeIcon + ' ' + escapeHtmlLf(trace.name) + '</div>';
    html += '<div style="display:flex; gap:8px;">';
    
    if (trace.score !== null && trace.score !== undefined) {
      html += '<span class="lf-score" style="font-size:0.65rem; padding:2px 6px; border-radius:3px; background:' + scoreColor + '20; color:' + scoreColor + ';">' + (trace.score * 100).toFixed(0) + '%</span>';
    }
    
    html += '<span style="font-size:0.65rem; color:' + latencyColor + ';">' + latencyStr + '</span>';
    html += '</div></div>';

    html += '<div style="display:flex; gap:12px; font-size:0.65rem; color:var(--text-dim);">';
    if (trace.total_tokens > 0) {
      html += '<span>' + trace.total_tokens.toLocaleString() + ' tokens</span>';
    }
    html += '<span>' + trace.timestamp + '</span>';
    html += '</div>';

    html += '</div>';
  });

  container.innerHTML = html;
}

function selectTrace(traceId) {
  selectedTraceId = traceId;
  
  const cards = document.querySelectorAll('.lf-trace-card');
  cards.forEach(function(card) {
    if (card.dataset.traceId === traceId) {
      card.style.borderColor = 'var(--amber)';
    } else {
      card.style.borderColor = 'var(--border)';
    }
  });

  renderTraceDetail(traceId);
  document.getElementById('lf-stage-2')?.classList.add('visible');
  document.getElementById('lf-stage-3')?.classList.add('visible');
}

function renderTraceDetail(traceId) {
  const container = document.getElementById('langfuse-detail');
  if (!container) return;

  const trace = SIMULATED_TRACES.find(function(t) { return t.id === traceId; });
  if (!trace) {
    container.innerHTML = '<span style="color:var(--text-dim);">Trace not found</span>';
    return;
  }

  let html = '';

  // Header
  html += '<div style="background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px; margin-bottom:10px;">';
  
  html += '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">';
  html += '<div style="font-size:0.85rem; font-weight:700; color:var(--amber);">' + escapeHtmlLf(trace.name) + '</div>';
  
  if (trace.score !== null && trace.score !== undefined) {
    const scoreColor = trace.score >= 0.85 ? 'var(--green)' :
                       trace.score >= 0.70 ? 'var(--amber)' : 'var(--red)';
    html += '<span style="font-size:0.75rem; font-weight:600; color:' + scoreColor + ';">Score: ' + (trace.score * 100).toFixed(0) + '% (' + trace.score_label + ')</span>';
  }
  html += '</div>';

  // Metrics row
  html += '<div style="display:flex; gap:16px; font-size:0.7rem; color:var(--text-sec); flex-wrap:wrap;">';
  
  const latencyStr = trace.latency_ms >= 1000 ? 
                     (trace.latency_ms / 1000).toFixed(2) + 's' : 
                     trace.latency_ms + 'ms';
  html += '<div><span style="color:var(--text-dim);">Latency:</span> <strong style="color:var(--white);">' + latencyStr + '</strong></div>';
  
  if (trace.total_tokens > 0) {
    html += '<div><span style="color:var(--text-dim);">Tokens:</span> <strong style="color:var(--white);">' + trace.total_tokens.toLocaleString() + '</strong>';
    html += ' <span style="color:var(--text-dim);">(' + trace.prompt_tokens + '/' + trace.completion_tokens + ')</span></div>';
  }
  
  html += '<div><span style="color:var(--text-dim);">Session:</span> <code style="color:var(--cyan); font-size:0.65rem;">' + escapeHtmlLf(trace.session_id) + '</code></div>';
  html += '</div>';

  // Metadata
  if (trace.metadata && Object.keys(trace.metadata).length > 0) {
    html += '<div style="margin-top:8px; padding-top:8px; border-top:1px solid var(--border); font-size:0.65rem;">';
    html += '<span style="color:var(--text-dim);">Metadata:</span> ';
    const metaParts = Object.entries(trace.metadata).map(function(kv) {
      return kv[0] + '=' + kv[1];
    });
    html += '<span style="color:var(--text-sec);">' + escapeHtmlLf(metaParts.join(', ')) + '</span>';
    html += '</div>';
  }

  html += '</div>';

  // Steps timeline
  if (trace.steps && trace.steps.length > 0) {
    html += '<div style="margin-bottom:10px;">';
    html += '<div style="font-size:0.75rem; font-weight:700; color:var(--cyan); margin-bottom:6px;">Execution Timeline</div>';
    
    let cumulativeLatency = 0;
    const totalLatency = trace.steps.reduce(function(sum, s) { return sum + (s.latency_ms || 0); }, 0);
    
    trace.steps.forEach(function(step, idx) {
      const pct = ((step.latency_ms || 0) / Math.max(totalLatency, 1)) * 100;
      cumulativeLatency += step.latency_ms || 0;
      
      const barColor = idx % 3 === 0 ? 'var(--orange)' : 
                       idx % 3 === 1 ? 'var(--cyan)' : 'var(--indigo)';
      
      html += '<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">';
      html += '<span style="font-size:0.65rem; color:var(--text-dim); width:120px; text-align:right;">' + escapeHtmlLf(step.name) + '</span>';
      html += '<div style="flex:1; height:8px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden;">';
      html += '<div class="lf-waterfall-bar" data-width="' + pct.toFixed(1) + '" style="width:0%; height:100%; background:' + barColor + '; opacity:0.7; border-radius:4px;"></div>';
      html += '</div>';
      
      const stepLat = (step.latency_ms || 0) >= 1000 ? 
                      ((step.latency_ms || 0) / 1000).toFixed(1) + 's' : 
                      (step.latency_ms || 0) + 'ms';
      html += '<span style="font-size:0.65rem; color:var(--text-sec); width:50px;">' + stepLat + '</span>';
      
      if (step.tokens > 0) {
        html += '<span style="font-size:0.6rem; color:var(--text-dim);">' + step.tokens + ' tok</span>';
      }
      html += '</div>';
    });
    
    html += '</div>';
  }

  // Prompt/Generation
  if (trace.prompt || trace.generation) {
    html += '<div style="background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:10px;">';
    
    if (trace.prompt) {
      html += '<div style="margin-bottom:8px;">';
      html += '<div style="font-size:0.7rem; font-weight:700; color:var(--indigo); margin-bottom:4px;">Prompt</div>';
      html += '<pre style="font-size:0.65rem; color:var(--text-sec); background:rgba(255,255,255,0.02); padding:6px; border-radius:4px; overflow-x:auto; white-space:pre-wrap; word-break:break-word; margin:0;">' + escapeHtmlLf(trace.prompt) + '</pre>';
      html += '</div>';
    }
    
    if (trace.generation) {
      html += '<div>';
      html += '<div style="font-size:0.7rem; font-weight:700; color:var(--green); margin-bottom:4px;">Generation</div>';
      html += '<pre style="font-size:0.65rem; color:var(--text-sec); background:rgba(255,255,255,0.02); padding:6px; border-radius:4px; overflow-x:auto; white-space:pre-wrap; word-break:break-word; margin:0;">' + escapeHtmlLf(trace.generation) + '</pre>';
      html += '</div>';
    }
    
    html += '</div>';
  }

  // Observations
  if (trace.observations && trace.observations.length > 0) {
    html += '<div style="margin-top:10px; background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:10px;">';
    html += '<div style="font-size:0.7rem; font-weight:700; color:var(--amber); margin-bottom:6px;">Observations</div>';
    html += '<div style="display:flex; gap:12px; flex-wrap:wrap;">';
    
    trace.observations.forEach(function(obs) {
      html += '<div style="font-size:0.65rem;">';
      html += '<span style="color:var(--text-dim);">' + escapeHtmlLf(obs.name) + ':</span> ';
      html += '<strong style="color:var(--white);">' + escapeHtmlLf(String(obs.value)) + '</strong>';
      html += '</div>';
    });
    
    html += '</div></div>';
  }

  container.innerHTML = html;

  // Animate waterfall bars after DOM update
  requestAnimationFrame(function() { animateWaterfallBars(); });
}

function renderScoreDashboard() {
  const container = document.getElementById('langfuse-scores');
  if (!container) return;

  const scoreTypes = [
    { label: 'Accuracy', avg: 0.89, min: 0.82, max: 0.95, count: 12 },
    { label: 'Relevance', avg: 0.78, min: 0.65, max: 0.91, count: 8 },
    { label: 'Convergence', avg: 0.92, min: 0.88, max: 0.97, count: 5 },
    { label: 'Helpfulness', avg: 0.81, min: 0.72, max: 0.89, count: 15 }
  ];

  let html = '';

  scoreTypes.forEach(function(s, idx) {
    const color = s.avg >= 0.85 ? 'var(--green)' : s.avg >= 0.70 ? 'var(--amber)' : 'var(--red)';

    html += '<div style="background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px; min-width:140px;">';

    html += '<div style="font-size:0.75rem; font-weight:700; color:' + color + ';">' + s.label + '</div>';
    html += '<div class="lf-score-counter" data-target="' + (s.avg * 100).toFixed(0) + '" style="font-size:1.5rem; font-weight:800; color:var(--white); margin-top:4px;">0%</div>';

    html += '<div style="display:flex; justify-content:space-between; font-size:0.6rem; color:var(--text-dim); margin-top:6px;">';
    html += '<span>min ' + (s.min * 100).toFixed(0) + '%</span>';
    html += '<span>max ' + (s.max * 100).toFixed(0) + '%</span>';
    html += '</div>';

    html += '<div style="font-size:0.6rem; color:var(--text-dim); margin-top:4px;">' + s.count + ' traces</div>';

    html += '</div>';
  });

  container.innerHTML = html;

  // Animated counter
  setTimeout(function() { animateScoreCounters(); }, 300);
}

function animateScoreCounters() {
  var counters = document.querySelectorAll('.lf-score-counter');
  counters.forEach(function(el, idx) {
    var target = parseInt(el.dataset.target, 10) || 0;
    var current = 0;
    var duration = 1500;
    var startTime = performance.now() + idx * 150;

    function step(now) {
      var elapsed = now - startTime;
      if (elapsed < 0) { requestAnimationFrame(step); return; }
      var progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      var eased = 1 - Math.pow(1 - progress, 3);
      current = Math.round(eased * target);
      el.textContent = current + '%';
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  });
}

/* ── Waterfall timeline animation for trace steps ── */
function animateWaterfallBars() {
  var bars = document.querySelectorAll('.lf-waterfall-bar');
  bars.forEach(function(bar, i) {
    bar.style.width = '0%';
    setTimeout(function() {
      bar.style.transition = 'width 0.4s cubic-bezier(.4,0,.2,1)';
      bar.style.width = bar.dataset.width + '%';
    }, 100 + i * 120);
  });
}

function escapeHtmlLf(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

registerAnim(7,
  function enter() {
    checkLangfuseHealth();
    renderTraceList();
    renderScoreDashboard();
    document.getElementById('lf-stage-1')?.classList.add('visible');
  },
  function leave() {
    selectedTraceId = null;
  }
);