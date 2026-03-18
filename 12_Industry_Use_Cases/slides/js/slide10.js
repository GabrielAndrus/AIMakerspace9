/* ══════════════════════════════════════════════════════════════
   SLIDE 10: LIVE ERROR INVESTIGATION AGENT DEMO (slide index 9)
   ══════════════════════════════════════════════════════════════ */

const SEARXNG = 'http://192.168.1.36:4000';

let currentError = null;
let investigationStep = 0;

const ERROR_SCENARIOS = {
  cuda_oom: {
    type: 'RuntimeError',
    message: 'CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total)',
    traceback: `Traceback (most recent call last):
  File "train.py", line 142, in <module>
    trainer.train()
  File "transformers/trainer.py", line 1537, in train
    loss.backward()
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB`,
    kb_results: [
      { title: 'CUDA OOM Fix - Gradient Checkpointing', source: 'automl_knowledge', score: 0.92, content: 'Enable gradient_checkpointing=True in training args to reduce memory by recomputing activations during backward pass.' },
      { title: 'Batch Size Reduction', source: 'error_investigations', score: 0.88, content: 'Reduce per_device_train_batch_size from 32 to 8 or lower. Use gradient_accumulation_steps to maintain effective batch size.' }
    ],
    web_results: [
      { title: 'PyTorch CUDA Memory Management', source: 'stackoverflow', url: 'https://stackoverflow.com/questions/59573547', content: 'Use torch.cuda.empty_cache() between operations and monitor with nvidia-smi.' },
      { title: 'HuggingFace Trainer Memory Optimization', source: 'huggingface', url: 'https://huggingface.co/docs/trainer/memory', content: 'Enable Flash Attention 2, use bfloat16 instead of float32, set optim="adamw_8bit" or "adafactor".' }
    ],
    fix: {
      confidence: 'high',
      steps: [
        { action: 'Reduce batch size', code: 'per_device_train_batch_size = 4' },
        { action: 'Enable gradient checkpointing', code: 'gradient_checkpointing = True' },
        { action: 'Use memory-efficient optimizer', code: 'optim = "adafactor"' },
        { action: 'Clear cache before training', code: 'torch.cuda.empty_cache()' }
      ],
      explanation: 'The GPU ran out of memory during backward pass. Reducing batch size and enabling gradient checkpointing are the most effective fixes.'
    }
  },
  nan_loss: {
    type: 'ValueError',
    message: 'Loss became NaN during training at step 234',
    traceback: `Traceback (most recent call last):
  File "train.py", line 156, in <module>
    trainer.train()
  File "transformers/trainer.py", line 1645, in train
    self._maybe_log_save()
ValueError: Loss is NaN - training diverged`,
    kb_results: [
      { title: 'NaN Loss Prevention', source: 'automl_knowledge', score: 0.94, content: 'Lower learning rate by 10x (e.g., from 1e-4 to 1e-5). Enable gradient clipping with max_grad_norm=0.3.' },
      { title: 'Gradient Clipping Configuration', source: 'error_investigations', score: 0.85, content: 'Set max_grad_norm in TrainingArguments to prevent gradient explosion. Values between 0.3 and 1.0 work well for LLM training.' }
    ],
    web_results: [
      { title: 'Debugging NaN Loss in Transformers', source: 'github', url: 'https://github.com/huggingface/transformers/issues/1234', content: 'Common causes: learning rate too high, bad data samples (empty strings), mixed precision issues.' },
      { title: 'Learning Rate Scheduler Best Practices', source: 'huggingface', url: 'https://huggingface.co/docs/transformers/scheduler', content: 'Use warmup_steps = 5-10% of total steps. Use cosine or linear scheduler for stable training.' }
    ],
    fix: {
      confidence: 'high',
      steps: [
        { action: 'Reduce learning rate', code: 'learning_rate = 1e-5  # was 1e-4' },
        { action: 'Enable gradient clipping', code: 'max_grad_norm = 0.3' },
        { action: 'Add warmup steps', code: 'warmup_steps = 100' },
        { action: 'Validate dataset for NaN/Inf', code: 'df.isna().sum() # check for nulls' }
      ],
      explanation: 'NaN loss typically indicates gradient explosion. Lower learning rate and gradient clipping are the primary fixes.'
    }
  },
  dim_mismatch: {
    type: 'RuntimeError',
    message: 'mat1 and mat2 shapes cannot be multiplied (64x768) and (1024x256)',
    traceback: `Traceback (most recent call last):
  File "model.py", line 89, in forward
    x = self.fc(x)
RuntimeError: mat1 and mat2 shapes cannot be multiplied`,
    kb_results: [
      { title: 'Tensor Shape Debugging', source: 'automl_knowledge', score: 0.91, content: 'Use torchsummary to print layer dimensions. Check hidden_size matches model config.' },
      { title: 'Linear Layer Dimension Fix', source: 'error_investigations', score: 0.82, content: 'Ensure in_features of linear layer matches the output dimension of previous layer.' }
    ],
    web_results: [
      { title: 'PyTorch Shape Mismatch Debugging', source: 'stackoverflow', url: 'https://stackoverflow.com/questions/54340598', content: 'Print tensor shapes at each step with print(x.shape). Check model config hidden_size.' },
      { title: 'HuggingFace Model Dimension Config', source: 'huggingface', url: 'https://huggingface.co/docs/transformers/model_config', content: 'Verify hidden_size, num_attention_heads, and intermediate_size match the pretrained model.' }
    ],
    fix: {
      confidence: 'medium',
      steps: [
        { action: 'Print tensor shapes', code: 'print(f"Input shape: {x.shape}")' },
        { action: 'Check config dimensions', code: 'config.hidden_size == 768?' },
        { action: 'Fix linear layer input', code: 'self.fc = nn.Linear(768, 256)' }
      ],
      explanation: 'Shape mismatch between layers. The input tensor has wrong dimension - check model config and adjust layer sizes.'
    }
  },
  keyerror_labels: {
    type: 'KeyError',
    message: "KeyError: 'labels' not found in batch dictionary",
    traceback: `Traceback (most recent call last):
  File "trainer.py", line 234, in training_step
    labels = batch["labels"]
KeyError: 'labels'`,
    kb_results: [
      { title: 'SFT Data Format Requirements', source: 'automl_knowledge', score: 0.96, content: 'TRL SFT trainer expects messages format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}' },
      { title: 'Dataset Column Mapping', source: 'error_investigations', score: 0.89, content: 'Use dataset.map() to rename columns or format data. SFTDataCollator needs specific fields.' }
    ],
    web_results: [
      { title: 'TRL SFT Trainer Data Format', source: 'huggingface', url: 'https://huggingface.co/docs/trl/sft_trainer', content: 'Dataset must have "messages" field with role/content pairs. Use format_dataset() helper.' },
      { title: 'Debugging KeyError in DataLoader', source: 'stackoverflow', url: 'https://stackoverflow.com/questions/45678901', content: 'Print batch keys to debug: print(batch.keys()). Check collator configuration.' }
    ],
    fix: {
      confidence: 'high',
      steps: [
        { action: 'Validate JSONL format', code: '!head -n 3 train.jsonl | jq .' },
        { action: 'Check dataset columns', code: 'print(dataset.column_names)' },
        { action: 'Format for SFT', code: 'dataset = dataset.map(format_messages)' }
      ],
      explanation: 'DataLoader expects "labels" or "messages" field. Check data collator settings and dataset format.'
    }
  },
  gradient_explosion: {
    type: 'RuntimeError',
    message: 'Gradient overflow detected - gradients contain inf values',
    traceback: `Traceback (most recent call last):
  File "train.py", line 201, in <module>
    trainer.train()
RuntimeError: Gradient overflow - detected inf/NaN in gradients`,
    kb_results: [
      { title: 'Mixed Precision Training Stability', source: 'automl_knowledge', score: 0.93, content: 'Use fp16 or bf16 mixed precision with loss scaling. Set fp16_backend="auto" or use native amp.' },
      { title: 'Gradient Overflow Fix', source: 'error_investigations', score: 0.87, content: 'Enable gradient clipping and reduce learning rate. Check for data issues causing extreme values.' }
    ],
    web_results: [
      { title: 'NVIDIA AMP Best Practices', source: 'nvidia', url: 'https://docs.nvidia.com/deeplearning/performance/', content: 'Use GradScaler with initial_scale=2^15. Monitor for overflow and adjust scale automatically.' },
      { title: 'HuggingFace Trainer Overflow Handling', source: 'github', url: 'https://github.com/huggingface/transformers/issues/5678', content: 'Set fp16=true, fp16_opt_level="O1". Enable gradient clipping with max_grad_norm=1.0.' }
    ],
    fix: {
      confidence: 'high',
      steps: [
        { action: 'Enable mixed precision', code: 'fp16 = True  # or bf16 = True' },
        { action: 'Set gradient clipping', code: 'max_grad_norm = 1.0' },
        { action: 'Reduce learning rate', code: 'learning_rate = 2e-5' },
        { action: 'Check data for outliers', code: 'df.describe().max() # look for inf/NaN' }
      ],
      explanation: 'Gradient overflow from mixed precision or high learning rate. Enable AMP with proper scaling and gradient clipping.'
    }
  }
};

function selectError(errorKey) {
  currentError = ERROR_SCENARIOS[errorKey];
  if (!currentError) return;

  const detailsEl = document.getElementById('error-details');
  const tracebackEl = document.getElementById('error-traceback');

  if (detailsEl && tracebackEl) {
    detailsEl.style.display = 'block';
    tracebackEl.textContent = currentError.traceback;
  }

  showErrOutput('<span style="color:var(--amber);">Error loaded. Click "Investigate" to start.</span>');

  const btnContainer = document.getElementById('err-stage-1').querySelector('.demo-stage-body');
  let btn = document.getElementById('btn-start-investigation');
  if (!btn) {
    btn = document.createElement('button');
    btn.id = 'btn-start-investigation';
    btn.className = 'btn btn-red btn-sm';
    btn.textContent = 'Investigate Error';
    btn.style.marginTop = '10px';
    btn.onclick = runErrorInvestigation;
    btnContainer.appendChild(btn);
  }
}

function showErrOutput(html) {
  const el = document.getElementById('err-out-1');
  if (!el) {
    const outDiv = document.createElement('div');
    outDiv.id = 'err-out-1';
    outDiv.className = 'demo-stage-output';
    outDiv.style.marginTop = '8px';
    outDiv.innerHTML = html;
    document.getElementById('err-stage-1').querySelector('.demo-stage-body').appendChild(outDiv);
  } else {
    el.style.display = 'block';
    el.innerHTML = html;
  }
}

async function runErrorInvestigation() {
  if (!currentError) {
    showErrOutput('<span style="color:var(--red);">Please select an error first.</span>');
    return;
  }

  const btn = document.getElementById('btn-start-investigation');
  setLoading('btn-start-investigation', true);

  investigationStep = 0;

  document.getElementById('err-stage-2')?.classList.add('visible');

  await animateInvestigation();

  setLoading('btn-start-investigation', false);
}

async function animateInvestigation() {
  const progressAnalyze = document.getElementById('progress-analyze');
  const progressKb = document.getElementById('progress-kb');
  const progressWeb = document.getElementById('progress-web');
  const progressEval = document.getElementById('progress-eval');

  if (progressAnalyze) {
    progressAnalyze.innerHTML = '✓ Analyzing error context...';
    progressAnalyze.style.color = '#b8bb26';
  }

  await delay(800);

  if (progressKb) {
    progressKb.style.display = 'block';
    progressKb.style.color = '#fabd2f';
    progressKb.innerHTML = '⏳ Searching knowledge base (Qdrant)...';
  }
  await delay(1200);
  if (progressKb && currentError && currentError.kb_results) {
    progressKb.innerHTML = '✓ Knowledge base: Found ' + currentError.kb_results.length + ' relevant documents';
    progressKb.style.color = '#8ec07c';
  }

  renderSearchResults();

  await delay(600);

  if (progressWeb) {
    progressWeb.style.display = 'block';
    progressWeb.style.color = '#fabd2f';
    progressWeb.innerHTML = '⏳ Querying web search (SearXNG @ :4000)...';
  }
  await delay(1500);
  if (progressWeb && currentError && currentError.web_results) {
    progressWeb.innerHTML = '✓ Web search: Found ' + currentError.web_results.length + ' additional sources';
    progressWeb.style.color = '#8ec07c';
  }

  document.getElementById('err-stage-3')?.classList.add('visible');
  await delay(500);
  renderSolutionEvaluation();

  if (progressEval) {
    progressEval.style.display = 'block';
    progressEval.style.color = '#fabd2f';
    progressEval.innerHTML = '⏳ Evaluating solution relevance...';
  }
  await delay(1000);
  if (progressEval) {
    progressEval.innerHTML = '✓ LLM filtered to top solutions with high confidence';
    progressEval.style.color = '#b8bb26';
  }

  document.getElementById('err-stage-4')?.classList.add('visible');
  await delay(300);
  renderFixInstructions();
}

function renderSearchResults() {
  const container = document.getElementById('search-results');
  if (!container || !currentError) return;

  let html = '<div style="font-size:0.72rem; font-weight:700; color:var(--cyan); margin-bottom:6px;">Qdrant Knowledge Base Results</div>';

  if (currentError.kb_results) {
    currentError.kb_results.forEach(function(r, i) {
      const scorePct = (r.score * 100).toFixed(0);
      html += '<div style="background:rgba(34,211,238,0.05); border-left:2px solid var(--cyan); padding:6px 8px; margin-bottom:4px; font-size:0.68rem;">';
      html += '<strong style="color:var(--white);">' + escapeHtml(r.title) + '</strong>';
      html += ' <span style="color:var(--green); float:right;">' + scorePct + '%</span><br>';
      html += '<span style="color:var(--text-sec);">' + escapeHtml(r.content.slice(0, 100)) + '\u2026</span>';
      html += '</div>';
    });
  }

  html += '<div style="font-size:0.72rem; font-weight:700; color:var(--amber); margin-top:10px; margin-bottom:6px;">SearXNG Web Results</div>';

  if (currentError.web_results) {
    currentError.web_results.forEach(function(r, i) {
      html += '<div style="background:rgba(245,158,11,0.05); border-left:2px solid var(--amber); padding:6px 8px; margin-bottom:4px; font-size:0.68rem;">';
      html += '<strong style="color:var(--white);">' + escapeHtml(r.title) + '</strong>';
      html += ' <span style="color:var(--text-dim);">(' + r.source + ')</span><br>';
      html += '<span style="color:var(--text-sec);">' + escapeHtml(r.content.slice(0, 80)) + '\u2026</span>';
      html += '</div>';
    });
  }

  container.innerHTML = html;
}

function renderSolutionEvaluation() {
  const container = document.getElementById('solution-eval');
  if (!container || !currentError) return;

  let html = '<div style="margin-bottom:8px;">';
  html += '<strong style="color:var(--text);">LLM Evaluation Results:</strong>';
  html += '</div>';

  if (currentError.kb_results) {
    currentError.kb_results.forEach(function(r, i) {
      const qualityLabel = r.score > 0.9 ? 'HIGHLY RELEVANT' : 'RELEVANT';
      const qualityColor = r.score > 0.9 ? 'var(--green)' : 'var(--amber)';
      html += '<div style="background:rgba(255,255,255,0.02); border-radius:4px; padding:6px 8px; margin-bottom:6px;">';
      html += '<span style="color:' + qualityColor + '; font-weight:600;">' + qualityLabel + '</span> ';
      html += '<span style="color:var(--text-dim);">— ' + escapeHtml(r.title) + '</span>';
      html += '</div>';
    });
  }

  if (currentError.web_results && currentError.fix) {
    currentError.web_results.slice(0, 2).forEach(function(r, i) {
      html += '<div style="background:rgba(255,255,255,0.02); border-radius:4px; padding:6px 8px; margin-bottom:6px;">';
      html += '<span style="color:var(--cyan);">FETCHED</span> ';
      html += '<span style="color:var(--text-dim);">— ' + escapeHtml(r.title) + '</span>';
      html += '</div>';
    });
  }

  if (currentError.fix) {
    html += '<div style="margin-top:10px; padding:8px; background:rgba(184,187,38,0.1); border-radius:6px;">';
    html += '<span style="color:#b8bb26; font-weight:700;">✓ Confidence Level: ' + currentError.fix.confidence.toUpperCase() + '</span>';
    html += '</div>';
  }

  container.innerHTML = html;
}

function renderFixInstructions() {
  const container = document.getElementById('fix-instructions');
  if (!container || !currentError || !currentError.fix) return;

  let html = '<div style="margin-bottom:8px;"><strong style="color:#b8bb26;">Recommended Fix:</strong></div>';
  html += '<p style="color:#a89984; font-size:0.7rem; margin:0 0 10px 0;">' + escapeHtml(currentError.fix.explanation) + '</p>';

  html += '<div style="font-size:0.72rem; font-weight:700; color:#ebdbb2; margin-bottom:6px;">Step-by-step:</div>';
  html += '<pre style="background:#1d2021; padding:10px; border-radius:4px; overflow-x:auto; white-space:pre-wrap;">';

  if (currentError.fix.steps) {
    currentError.fix.steps.forEach(function(step, i) {
      html += '<span style="color:#fabd2f;"># ' + (i + 1) + '. ' + escapeHtml(step.action) + '</span>\n';
      html += '<span style="color:#8ec07c;">' + escapeHtml(step.code) + '</span>\n\n';
    });
  }

  html += '</pre>';

  html += '<div style="margin-top:10px;">';
  html += '<button class="btn btn-green btn-sm" onclick="retryWithFix()" style="margin-right:8px;">Apply Fix & Retry</button>';
  html += '<span style="color:var(--text-dim); font-size:0.65rem;">Agent will modify config and re-run training</span>';
  html += '</div>';

  container.innerHTML = html;
}

function retryWithFix() {
  const fixEl = document.getElementById('fix-instructions');
  if (fixEl) {
    const originalBg = fixEl.style.background;
    fixEl.style.background = 'rgba(34,197,94,0.2)';
    setTimeout(function() {
      fixEl.innerHTML = '<div style="color:var(--green); font-weight:700;">✓ Fix applied successfully!</div>' +
        '<p style="color:var(--text-sec); font-size:0.7rem; margin-top:8px;">Training reconfigured with optimized settings.</p>' +
        '<pre style="background:rgba(0,0,0,0.3); padding:10px; border-radius:4px; margin-top:8px; font-size:0.68rem;">' +
        'trainer = Trainer(\n  model=model,\n  args=TrainingArguments(\n    per_device_train_batch_size=4,\n    gradient_checkpointing=True,\n    max_grad_norm=0.3\n  )\n)\ntrainer.train()  # ✓ Success</pre>';
    }, 800);
  }
}

/* delay, setLoading, and escapeHtml are now in engine.js */

function delay(ms) {
  return new Promise(function(resolve) { setTimeout(resolve, ms); });
}

/* ── Agent Brain Canvas — abstract neural network visualization ── */
var brainAnimFrame = null;
var brainNodes = [];
var brainEdges = [];
var brainPulses = [];
var brainPhase = 'idle'; // idle, analyzing, searching, evaluating, resolved

function initBrainNetwork() {
  brainNodes = [];
  brainEdges = [];
  brainPulses = [];

  // Generate nodes in a horizontal band
  var count = 18;
  for (var i = 0; i < count; i++) {
    brainNodes.push({
      x: 40 + (i / (count - 1)) * 880,
      y: 15 + Math.sin(i * 0.8) * 15 + Math.random() * 10,
      r: 2.5 + Math.random() * 2
    });
  }

  // Connect nearby nodes
  for (var a = 0; a < count; a++) {
    for (var b = a + 1; b < count; b++) {
      var dx = brainNodes[a].x - brainNodes[b].x;
      var dy = brainNodes[a].y - brainNodes[b].y;
      if (Math.sqrt(dx * dx + dy * dy) < 120) {
        brainEdges.push([a, b]);
      }
    }
  }
}

function getBrainColor() {
  if (brainPhase === 'analyzing') return '#f59e0b';
  if (brainPhase === 'searching') return '#22d3ee';
  if (brainPhase === 'evaluating') return '#f59e0b';
  if (brainPhase === 'resolved') return '#10b981';
  return 'rgba(235,219,178,0.15)';
}

function drawBrainCanvas(t) {
  var canvas = document.getElementById('agent-brain-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  var color = getBrainColor();
  var active = brainPhase !== 'idle';

  // Draw edges
  for (var e = 0; e < brainEdges.length; e++) {
    var n1 = brainNodes[brainEdges[e][0]];
    var n2 = brainNodes[brainEdges[e][1]];
    ctx.strokeStyle = active ? color : 'rgba(235,219,178,0.06)';
    ctx.globalAlpha = active ? 0.15 : 0.06;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(n1.x, n1.y);
    ctx.lineTo(n2.x, n2.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Spawn pulses when active
  if (active && Math.random() < 0.08) {
    var ei = Math.floor(Math.random() * brainEdges.length);
    brainPulses.push({ edge: ei, progress: 0 });
  }

  // Draw pulses
  for (var p = brainPulses.length - 1; p >= 0; p--) {
    var pulse = brainPulses[p];
    pulse.progress += 0.025;
    if (pulse.progress > 1) { brainPulses.splice(p, 1); continue; }

    var pe = brainEdges[pulse.edge];
    var pn1 = brainNodes[pe[0]], pn2 = brainNodes[pe[1]];
    var px = pn1.x + (pn2.x - pn1.x) * pulse.progress;
    var py = pn1.y + (pn2.y - pn1.y) * pulse.progress;

    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.6;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // Draw nodes
  for (var i = 0; i < brainNodes.length; i++) {
    var n = brainNodes[i];
    var breathe = active ? Math.sin(t * 0.003 + i) * 0.3 + 0.7 : 0.3;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = active ? color : 'rgba(235,219,178,0.2)';
    ctx.globalAlpha = breathe;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

function animateBrain() {
  drawBrainCanvas(performance.now());
  brainAnimFrame = requestAnimationFrame(animateBrain);
}

/* ── Ambient pulse on agent progress terminal ── */
var ambientPulseInterval = null;

function startAmbientPulse() {
  var el = document.getElementById('agent-progress');
  if (!el) return;
  ambientPulseInterval = setInterval(function() {
    if (brainPhase === 'idle' || brainPhase === 'resolved') {
      el.style.boxShadow = 'none';
      return;
    }
    var color = brainPhase === 'searching' ? '34,211,238' : '245,158,11';
    el.style.boxShadow = '0 0 12px rgba(' + color + ',0.08)';
    setTimeout(function() {
      el.style.boxShadow = '0 0 20px rgba(' + color + ',0.15)';
    }, 1000);
  }, 2000);
}

function stopAmbientPulse() {
  if (ambientPulseInterval) {
    clearInterval(ambientPulseInterval);
    ambientPulseInterval = null;
  }
  var el = document.getElementById('agent-progress');
  if (el) el.style.boxShadow = 'none';
}

/* ── Override investigation to sync brain phases ── */
var _originalAnimateInvestigation = animateInvestigation;
animateInvestigation = async function() {
  brainPhase = 'analyzing';
  startAmbientPulse();

  var progressAnalyze = document.getElementById('progress-analyze');
  var progressKb = document.getElementById('progress-kb');
  var progressWeb = document.getElementById('progress-web');
  var progressEval = document.getElementById('progress-eval');

  if (progressAnalyze) {
    progressAnalyze.innerHTML = '\u2713 Analyzing error context...';
    progressAnalyze.style.color = '#b8bb26';
  }

  await delay(800);
  brainPhase = 'searching';

  if (progressKb) {
    progressKb.style.display = 'block';
    progressKb.style.color = '#fabd2f';
    progressKb.innerHTML = '\u23F3 Searching knowledge base (Qdrant)...';
  }
  await delay(1200);
  if (progressKb && currentError && currentError.kb_results) {
    progressKb.innerHTML = '\u2713 Knowledge base: Found ' + currentError.kb_results.length + ' relevant documents';
    progressKb.style.color = '#8ec07c';
  }

  renderSearchResults();

  await delay(600);

  if (progressWeb) {
    progressWeb.style.display = 'block';
    progressWeb.style.color = '#fabd2f';
    progressWeb.innerHTML = '\u23F3 Querying web search (SearXNG @ :4000)...';
  }
  await delay(1500);
  if (progressWeb && currentError && currentError.web_results) {
    progressWeb.innerHTML = '\u2713 Web search: Found ' + currentError.web_results.length + ' additional sources';
    progressWeb.style.color = '#8ec07c';
  }

  brainPhase = 'evaluating';
  document.getElementById('err-stage-3')?.classList.add('visible');
  await delay(500);
  renderSolutionEvaluation();

  if (progressEval) {
    progressEval.style.display = 'block';
    progressEval.style.color = '#fabd2f';
    progressEval.innerHTML = '\u23F3 Evaluating solution relevance...';
  }
  await delay(1000);
  if (progressEval) {
    progressEval.innerHTML = '\u2713 LLM filtered to top solutions with high confidence';
    progressEval.style.color = '#b8bb26';
  }

  brainPhase = 'resolved';

  document.getElementById('err-stage-4')?.classList.add('visible');
  await delay(300);
  renderFixInstructionsWithTyping();
  stopAmbientPulse();
};

/* ── Fix instructions with typing effect ── */
function renderFixInstructionsWithTyping() {
  var container = document.getElementById('fix-instructions');
  if (!container || !currentError || !currentError.fix) {
    renderFixInstructions();
    return;
  }

  var lines = [];
  lines.push({ text: 'Recommended Fix:', color: '#b8bb26', bold: true });
  lines.push({ text: currentError.fix.explanation, color: '#a89984', bold: false });
  lines.push({ text: '', color: '', bold: false });

  if (currentError.fix.steps) {
    currentError.fix.steps.forEach(function(step, i) {
      lines.push({ text: '# ' + (i + 1) + '. ' + step.action, color: '#c9a84c', bold: false });
      lines.push({ text: step.code, color: '#8ec07c', bold: false });
    });
  }

  container.innerHTML = '';
  var pre = document.createElement('pre');
  pre.style.cssText = 'background:#1d2021; padding:10px; border-radius:4px; overflow-x:auto; white-space:pre-wrap; margin:0;';
  container.appendChild(pre);

  var lineIdx = 0;
  function typeLine() {
    if (lineIdx >= lines.length) {
      // Add apply button after typing
      var btnDiv = document.createElement('div');
      btnDiv.style.marginTop = '10px';
      btnDiv.innerHTML = '<button class="btn btn-green btn-sm" onclick="retryWithFix()" style="margin-right:8px;">Apply Fix & Retry</button>' +
        '<span style="color:var(--text-dim); font-size:0.65rem;">Agent will modify config and re-run training</span>';
      container.appendChild(btnDiv);
      return;
    }

    var line = lines[lineIdx];
    var span = document.createElement('span');
    span.style.color = line.color;
    if (line.bold) span.style.fontWeight = '700';

    pre.appendChild(span);

    var text = line.text;
    var charIdx = 0;

    function typeChar() {
      if (charIdx < text.length) {
        span.textContent += text[charIdx];
        charIdx++;
        setTimeout(typeChar, 12);
      } else {
        pre.appendChild(document.createTextNode('\n'));
        lineIdx++;
        setTimeout(typeLine, 50);
      }
    }

    if (text.length === 0) {
      pre.appendChild(document.createTextNode('\n'));
      lineIdx++;
      setTimeout(typeLine, 30);
    } else {
      typeChar();
    }
  }

  typeLine();
}

document.addEventListener('DOMContentLoaded', function() {
  var selectEl = document.getElementById('error-type-select');
  if (selectEl) {
    selectEl.addEventListener('change', function(e) {
      selectError(e.target.value);
    });
  }
});

registerAnim(9,
  function enter() {
    var statusDot = document.getElementById('err-status-dot');
    var statusText = document.getElementById('err-status-text');
    if (statusDot) { statusDot.classList.remove('down'); statusDot.classList.add('healthy'); }
    if (statusText) { statusText.textContent = 'Error Investigation Agent ready'; }

    document.getElementById('err-stage-1')?.classList.add('visible');

    var progressSteps = document.querySelectorAll('.agent-step');
    progressSteps.forEach(function(step) {
      step.style.display = 'none';
      step.style.color = 'var(--text-dim)';
    });
    var analyzeStep = document.getElementById('progress-analyze');
    if (analyzeStep) {
      analyzeStep.style.display = 'block';
      analyzeStep.innerHTML = '\u23F3 Analyzing error context...';
    }

    var resultsEl = document.getElementById('search-results');
    if (resultsEl) resultsEl.innerHTML = '';
    var evalEl = document.getElementById('solution-eval');
    if (evalEl) evalEl.innerHTML = '';
    var fixEl = document.getElementById('fix-instructions');
    if (fixEl) fixEl.innerHTML = '';

    currentError = null;
    investigationStep = 0;
    brainPhase = 'idle';

    var detailsEl = document.getElementById('error-details');
    if (detailsEl) detailsEl.style.display = 'none';
    var selectEl = document.getElementById('error-type-select');
    if (selectEl) selectEl.value = '';
    var outEl = document.getElementById('err-out-1');
    if (outEl) { outEl.style.display = 'none'; outEl.innerHTML = ''; }

    // Start brain animation
    initBrainNetwork();
    animateBrain();
  },
  function leave() {
    currentError = null;
    brainPhase = 'idle';
    stopAmbientPulse();
    if (brainAnimFrame) {
      cancelAnimationFrame(brainAnimFrame);
      brainAnimFrame = null;
    }
  }
);