/* ══════════════════════════════════════════════════════════════
   SLIDE 6: LIVE METAFLOW DEMO (slide index 5)
   ══════════════════════════════════════════════════════════════ */

const METAFLOW_PORT = 3001;

var MF_FLOWS = {
  MLTrainingFlow: {
    name: 'MLTrainingFlow',
    description: 'Tabular AutoML Pipeline',
    steps: [
      { name: 'start', desc: 'Initialize flow' },
      { name: 'load_data', desc: 'Load CSV into DataFrame', hasCard: true },
      { name: 'validate_data', desc: 'Validate structure & target' },
      { name: 'preprocess', desc: 'Detect task type, split data' },
      { name: 'train_model', desc: 'GridSearchCV ensemble' },
      { name: 'evaluate', desc: 'Test metrics & confusion matrix', hasCard: true },
      { name: 'save_model', desc: 'Serialize Pipeline + metadata' },
      { name: 'end', desc: 'Complete flow' }
    ],
    color: 'var(--orange)'
  },
  LLMTrainingFlow: {
    name: 'LLMTrainingFlow',
    description: 'LLM Fine-tuning (SFT/DPO/GRPO)',
    steps: [
      { name: 'start', desc: 'Initialize with parameters' },
      { name: 'load_data', desc: 'Load training file', hasCard: true },
      { name: 'validate_data', desc: 'Check file format' },
      { name: 'convert_data', desc: 'Convert to datasets format' },
      { name: 'train_model', desc: 'Fine-tune with LoRA', hasCard: true },
      { name: 'evaluate', desc: 'Generate metrics', hasCard: true },
      { name: 'save_model', desc: 'Save adapter package' },
      { name: 'end', desc: 'Complete flow' }
    ],
    color: 'var(--indigo)'
  }
};

var MF_SELECTED_FLOW = 'MLTrainingFlow';
var MF_RUN_COUNTER = 3;

/* setLoading is now in engine.js */

function showOutput(id, html) {
  var el = document.getElementById(id);
  if (!el) return;
  el.style.display = 'block';
  el.innerHTML = html;
}

async function checkMetaflowStatus() {
  var dot = document.getElementById('mf-status-dot');
  var text = document.getElementById('mf-status-text');
  
  try {
    var resp = await fetch('http://192.168.1.33:' + METAFLOW_PORT + '/', { 
      signal: AbortSignal.timeout(3000),
      mode: 'cors'
    });
    if (resp.ok) {
      if (dot) dot.classList.add('on');
      if (text) text.textContent = 'Metaflow Dashboard connected on :' + METAFLOW_PORT;
    } else {
      throw new Error('HTTP ' + resp.status);
    }
  } catch (e) {
    if (dot) { dot.classList.remove('on'); dot.classList.add('off'); }
    if (text) text.textContent = 'Demo mode — using simulated data';
  }
}

async function loadMetaflowFlow() {
  setLoading('btn-mf-stage-1', true);
  var select = document.getElementById('mf-flow-select');
  MF_SELECTED_FLOW = select ? select.value : 'MLTrainingFlow';
  
  showOutput('mf-out-1', '<span style="color:var(--amber);">Loading flow definition...</span>');
  await new Promise(function(r) { setTimeout(r, 600); });
  
  var flow = MF_FLOWS[MF_SELECTED_FLOW];
  if (!flow) {
    showOutput('mf-out-1', '<span style="color:var(--red);">Flow not found</span>');
    setLoading('btn-mf-stage-1', false);
    return;
  }
  
  var html = '<span style="color:var(--green);">✓ Flow loaded: ' + flow.name + '</span>';
  html += '<br><span style="color:var(--text-dim); font-size:0.68rem;">' + flow.steps.length + ' steps · ' + flow.description + '</span>';
  
  showOutput('mf-out-1', html);
  document.getElementById('mf-stage-1')?.classList.add('done');
  document.getElementById('mf-stage-2')?.classList.add('visible');
  
  renderPipelineVisualization(flow);
  
  setLoading('btn-mf-stage-1', false);
}

function renderPipelineVisualization(flow) {
  var container = document.getElementById('mf-pipeline-viz');
  if (!container) return;
  container.innerHTML = '';

  // Conveyor belt wrapper
  var belt = document.createElement('div');
  belt.className = 'mf-conveyor-belt';
  belt.style.cssText = 'position:relative; display:flex; align-items:center; gap:0; padding:8px 0;';

  // Conveyor belt track lines
  var track = document.createElement('div');
  track.className = 'mf-conveyor-track';
  track.style.cssText = 'position:absolute; left:14px; right:14px; top:50%; height:0; border-top:2px dashed rgba(235,219,178,0.12); animation: mf-belt-move 3s linear infinite;';
  belt.appendChild(track);

  // Package indicator (hidden initially, shown during run)
  var pkg = document.createElement('div');
  pkg.id = 'mf-package';
  pkg.style.cssText = 'position:absolute; left:0; top:50%; transform:translate(-50%,-50%); width:12px; height:12px; border-radius:3px; background:' + flow.color + '; opacity:0; z-index:2; transition:left 0.5s ease-in-out, opacity 0.3s; box-shadow:0 0 8px ' + flow.color + '40;';
  belt.appendChild(pkg);

  flow.steps.forEach(function(step, i) {
    var station = document.createElement('div');
    station.className = 'mf-station';
    station.id = 'mf-step-' + i;
    station.style.cssText = 'display:flex; flex-direction:column; align-items:center; gap:4px; flex:1; position:relative; z-index:1;';

    // Station card
    var card = document.createElement('div');
    card.className = 'mf-station-card';
    card.style.cssText = 'width:70px; padding:6px 4px; background:var(--surface); border:1.5px solid var(--border-hi); border-radius:6px; text-align:center; transition:all 0.3s;';

    var num = document.createElement('div');
    num.style.cssText = 'font-size:0.58rem; font-weight:700; color:var(--text-dim); font-family:var(--mono);';
    num.textContent = i + 1;
    card.appendChild(num);

    var name = document.createElement('div');
    name.style.cssText = 'font-size:0.58rem; color:var(--text-sec); margin-top:2px; line-height:1.3;';
    name.textContent = step.name.replace(/_/g, ' ');
    card.appendChild(name);

    if (step.hasCard) {
      var badge = document.createElement('div');
      badge.style.cssText = 'font-size:0.48rem; color:var(--gold); margin-top:3px;';
      badge.textContent = '@card';
      card.appendChild(badge);
    }

    station.appendChild(card);
    belt.appendChild(station);

    // Animate station appearance
    setTimeout(function() {
      card.style.borderColor = flow.color;
    }, i * 100 + 300);
  });

  container.appendChild(belt);

  var outHtml = '<span style="color:var(--green);">✓ Pipeline graph rendered</span>';
  showOutput('mf-out-2', outHtml);
  document.getElementById('mf-stage-2')?.classList.add('done');
  document.getElementById('mf-stage-3')?.classList.add('visible');

  renderRunsTable(flow);
}

function renderRunsTable(flow) {
  var container = document.getElementById('mf-runs-table');
  if (!container) return;
  
  var mockRuns = [
    { id: 'run_' + (Date.now() - 7200000), status: '✓', started: '2 hours ago', duration: '12.4s', user: 'demo' },
    { id: 'run_' + (Date.now() - 3600000), status: '✓', started: '1 hour ago', duration: '8.7s', user: 'demo' },
    { id: 'run_' + (Date.now() - 1800000), status: '✗', started: '30 min ago', duration: '—', user: 'demo', error: 'DataValidationError: Target column not found' }
  ];
  
  var html = '<div style="background:var(--bg); border:1px solid var(--border); border-radius:6px; overflow:hidden;">';
  html += '<div style="display:flex; background:var(--surface2); padding:8px 12px; font-weight:700; color:var(--text); border-bottom:1px solid var(--gold-border); font-size:0.68rem;">';
  html += '<span style="flex:2;">Run ID</span><span style="width:50px; text-align:center;">Status</span><span style="flex:1;">Started</span><span style="width:80px;">Duration</span></div>';

  mockRuns.forEach(function(run, idx) {
    var statusColor = run.status === '\u2713' ? 'var(--green)' : 'var(--red)';
    var rowBg = idx % 2 === 0 ? 'background:rgba(255,255,255,0.01);' : '';
    var borderLeft = run.status === '\u2713' ? 'border-left:2px solid var(--green);' : 'border-left:2px solid var(--red);';
    html += '<div style="display:flex; padding:6px 12px; border-bottom:1px solid var(--border); color:var(--text-sec); font-size:0.65rem; ' + rowBg + borderLeft + '">';
    html += '<span style="flex:2; font-family:var(--mono); color:var(--cyan);">' + run.id.slice(0, 18) + '...</span>';
    html += '<span style="width:50px; text-align:center; color:' + statusColor + '; font-weight:700;">' + run.status + '</span>';
    html += '<span style="flex:1;">' + run.started + '</span>';
    html += '<span style="width:80px;">' + run.duration + '</span></div>';
  });
  
  html += '</div>';
  container.innerHTML = html;
}

async function triggerMetaflowRun() {
  setLoading('btn-mf-trigger', true);
  var flow = MF_FLOWS[MF_SELECTED_FLOW];

  showOutput('mf-out-3', '<span style="color:var(--amber);">Starting ' + flow.name + '...</span>');

  // Show package
  var pkg = document.getElementById('mf-package');
  if (pkg) {
    pkg.style.opacity = '1';
    pkg.style.left = '0%';
  }

  for (var i = 0; i < flow.steps.length; i++) {
    await new Promise(function(r) { setTimeout(r, 400 + Math.random() * 300); });

    // Move package to this station
    var pct = ((i + 0.5) / flow.steps.length) * 100;
    if (pkg) pkg.style.left = pct + '%';

    var stepNode = document.getElementById('mf-step-' + i);
    if (stepNode) {
      var card = stepNode.querySelector('.mf-station-card');
      if (card) {
        // Light up station
        card.style.borderColor = '#b8bb26';
        card.style.boxShadow = '0 0 8px rgba(184,187,38,0.2)';
      }
      showOutput('mf-out-3', '<span style="color:var(--amber);">Running step: ' + flow.steps[i].name + '</span>');
    }

    // Brief pause at station
    await new Promise(function(r) { setTimeout(r, 200); });

    // Mark station done
    if (stepNode) {
      var card2 = stepNode.querySelector('.mf-station-card');
      if (card2) {
        card2.style.boxShadow = 'none';
      }
    }
  }

  // Hide package
  if (pkg) {
    pkg.style.left = '100%';
    setTimeout(function() { pkg.style.opacity = '0'; }, 400);
  }

  var newRunId = 'run_' + Date.now();
  MF_RUN_COUNTER++;

  var html = '<span style="color:var(--green);">✓ Run completed successfully</span>';
  html += '<br><span style="color:var(--cyan); font-size:0.68rem;">Run ID: ' + newRunId.slice(0, 18) + '...</span>';
  html += '<br><span style="color:var(--text-dim); font-size:0.65rem;">Duration: ' + (5 + Math.random() * 8).toFixed(1) + 's · All steps passed</span>';

  showOutput('mf-out-3', html);

  document.getElementById('mf-stage-3')?.classList.add('done');
  document.getElementById('mf-stage-4')?.classList.add('visible');

  renderMetaflowCards(flow);

  setLoading('btn-mf-trigger', false);
}

function renderMetaflowCards(flow) {
  var container = document.getElementById('mf-cards-display');
  if (!container) return;
  container.innerHTML = '';
  
  if (MF_SELECTED_FLOW === 'MLTrainingFlow') {
    var card1 = createCardEl('Model Metrics', [
      ['Accuracy', '87.3%'],
      ['F1 Macro', '0.856'],
      ['Precision', '0.841'],
      ['Recall', '0.872']
    ], 'var(--orange)');
    container.appendChild(card1);
    
    var card2 = createCardEl('Feature Importance', [
      ['Sex', '32%'],
      ['Fare', '18%'],
      ['Age', '15%'],
      ['Pclass', '14%']
    ], 'var(--cyan)');
    container.appendChild(card2);
    
    var card3 = createMarkdownCard('Best Model', 'XGBoostClassifier\nGridSearchCV best params:\nmax_depth=6, n_estimators=200');
    container.appendChild(card3);
  } else {
    var card1 = createCardEl('Training Config', [
      ['Base Model', 'Qwen2.5-0.5B'],
      ['Method', 'SFT'],
      ['Epochs', '3'],
      ['LR', '2e-4']
    ], 'var(--indigo)');
    container.appendChild(card1);
    
    var card2 = createCardEl('Adapter Info', [
      ['LoRA r', '16'],
      ['Target modules', 'q_proj, v_proj'],
      ['Trainable params', '1.2M'],
      ['GPU Memory', '4.8 GB']
    ], 'var(--green)');
    container.appendChild(card2);
    
    var card3 = createMarkdownCard('Output Path', 'models/llm_sft_adapter/\nadapter_config.json\nadapter_model.safetensors');
    container.appendChild(card3);
  }
}

function createCardEl(title, rows, accent) {
  var card = document.createElement('div');
  card.style.cssText = 'background:var(--surface); border:1px solid ' + accent.replace('var(--', 'rgba(var(--').replace(')', ',0.3)') + '; border-radius:8px; padding:12px; min-width:160px;';
  
  var header = document.createElement('div');
  header.style.cssText = 'font-size:0.75rem; font-weight:700; color:' + accent + '; margin-bottom:8px;';
  header.textContent = title;
  card.appendChild(header);
  
  rows.forEach(function(row) {
    var rowEl = document.createElement('div');
    rowEl.style.cssText = 'display:flex; justify-content:space-between; font-size:0.68rem; color:var(--text-sec); margin-bottom:2px;';
    rowEl.innerHTML = '<span>' + row[0] + '</span><span style="color:var(--text); font-family:monospace;">' + row[1] + '</span>';
    card.appendChild(rowEl);
  });
  
  return card;
}

function createMarkdownCard(title, content) {
  var card = document.createElement('div');
  card.style.cssText = 'background:var(--surface); border:1px solid var(--border-hi); border-radius:8px; padding:12px; min-width:200px;';
  
  var header = document.createElement('div');
  header.style.cssText = 'font-size:0.75rem; font-weight:700; color:var(--amber); margin-bottom:8px;';
  header.textContent = title;
  card.appendChild(header);
  
  var pre = document.createElement('pre');
  pre.style.cssText = 'font-family:monospace; font-size:0.65rem; color:var(--text-sec); white-space:pre-wrap; margin:0;';
  pre.textContent = content;
  card.appendChild(pre);
  
  return card;
}

function initMetaflowDemo() {
  checkMetaflowStatus();
  
  var stage1 = document.getElementById('mf-stage-1');
  if (stage1 && !stage1.classList.contains('visible')) {
    stage1.classList.add('visible');
  }
}

registerAnim(5,
  function enter() { initMetaflowDemo(); },
  null
);