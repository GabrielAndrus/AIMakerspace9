/* ══════════════════════════════════════════════════════════════
   SLIDE 9: LIVE LLM TRAINING DEMO — SFT/DPO/GRPO (slide index 8)
   ══════════════════════════════════════════════════════════════ */

var PRETRAIN_RESPONSE = "To build an agent, you first need to identify the principal — that is, the person or business on whose behalf the agent will act. An agent is authorized to represent another party, like a deputy or representative. You might consider hiring an FBI agent if government work is involved, or a literary agent if you need someone to manage publishing deals. In chemistry, an agent is a substance that causes a reaction, so you could build one in a laboratory. In pharmacology, an agent is a drug capable of eliciting a biological response. In grammar, the agent is the noun phrase that performs the action of the verb. To build any of these agents, consult the relevant professional licensing authority in your jurisdiction.";

let selectedMethod = null;
let trainingInterval = null;
let lossHistory = [];
let currentEpoch = 0;

const METHOD_INFO = {
  sft: {
    name: 'SFT — Supervised Fine-Tuning',
    icon: '📝',
    description: 'Train on instruction-response pairs. Best for teaching new tasks or domains.',
    datasetFormat: '{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}',
    metrics: ['training_loss', 'eval_loss'],
    color: '#a855f7'
  },
  dpo: {
    name: 'DPO — Direct Preference Optimization',
    icon: '⚖️',
    description: 'Align model using preference pairs (chosen vs rejected). No separate reward model needed.',
    datasetFormat: '{"prompt": [...], "chosen": [...], "rejected": [...]}',
    metrics: ['rewards/chosen', 'rewards/rejected', 'margin'],
    color: '#22d3ee'
  },
  grpo: {
    name: 'GRPO — Group Relative Policy Optimization',
    icon: '🎯',
    description: 'Group-based reward optimization with KL divergence constraints. Efficient RL training.',
    datasetFormat: '{"prompt": "...", "ground_truth": "...", "pattern": "regex"}',
    metrics: ['reward_mean', 'kl_divergence', 'score'],
    color: '#f59e0b'
  }
};

function selectTrainingMethod(method) {
  selectedMethod = method;
  
  document.querySelectorAll('.method-card').forEach(function(card) {
    card.classList.toggle('selected', card.dataset.method === method);
  });
  
  var select = document.getElementById('training-method');
  if (select) select.value = method;
  
  showMethodExplanation(method);
  
  var stage1 = document.getElementById('llm-stage-1');
  var stage2 = document.getElementById('llm-stage-2');
  if (stage1) stage1.classList.add('done');
  if (stage2) stage2.classList.add('visible');
}

function showMethodExplanation(method) {
  var container = document.getElementById('method-explanation');
  if (!container || !method) {
    if (container) container.style.display = 'none';
    return;
  }
  
  var info = METHOD_INFO[method];
  if (!info) return;
  
  container.innerHTML = 
    '<div style="background:var(--surface2); border-radius:8px; padding:12px; margin-top:8px; border-left:3px solid ' + info.color + ';">' +
    '<div style="font-size:0.78rem; font-weight:700; color:' + info.color + '; margin-bottom:4px;">' + info.name + '</div>' +
    '<div style="font-size:0.72rem; color:var(--text-sec);">' + info.description + '</div>' +
    '<div style="margin-top:8px; font-size:0.68rem; color:var(--text-dim);">Dataset format:</div>' +
    '<code style="display:block; margin-top:4px; font-size:0.62rem; background:var(--bg); padding:6px 10px; border-radius:4px; color:var(--green); white-space:pre-wrap; word-break:break-all;">' + info.datasetFormat + '</code>' +
    '</div>';
  container.style.display = 'block';
}

function startLLMTraining() {
  var method = selectedMethod || document.getElementById('training-method')?.value;
  if (!method) {
    alert('Please select a training method first.');
    return;
  }
  
  var stage2 = document.getElementById('llm-stage-2');
  var stage3 = document.getElementById('llm-stage-3');
  if (stage2) stage2.classList.add('done');
  if (stage3) stage3.classList.add('visible');
  
  runTrainingSimulation(method);
}

function runTrainingSimulation(method) {
  var info = METHOD_INFO[method];
  var epochs = parseInt(document.getElementById('train-epochs')?.value || '3', 10);
  var progressBar = document.getElementById('train-progress-bar');
  var progressPct = document.getElementById('train-progress-pct');
  var metricsContainer = document.getElementById('training-metrics');
  
  lossHistory = [];
  currentEpoch = 0;
  var totalSteps = epochs * 10;
  var step = 0;
  
  if (progressBar) progressBar.style.width = '0%';
  if (progressPct) progressPct.textContent = '0%';
  
  clearLossCanvas();
  
  var metricsHtml = '';
  info.metrics.forEach(function(m) {
    metricsHtml += '<div class="train-metric" id="metric-' + m.replace('/', '-') + '" style="background:var(--surface2); border-radius:8px; padding:10px 14px; min-width:120px;">' +
      '<div style="font-size:0.62rem; color:var(--text-dim); text-transform:uppercase;">' + m + '</div>' +
      '<div class="metric-value" style="font-size:1.1rem; font-weight:700; color:' + info.color + '; font-family:\'IBM Plex Mono\',monospace;">--</div>' +
    '</div>';
  });
  if (metricsContainer) metricsContainer.innerHTML = metricsHtml;
  
  trainingInterval = setInterval(function() {
    step++;
    
    var progress = Math.min(100, (step / totalSteps) * 100);
    if (progressBar) progressBar.style.width = progress + '%';
    if (progressPct) progressPct.textContent = Math.round(progress) + '%';
    
    var lossVal;
    if (method === 'sft') {
      lossVal = 2.5 - (step / totalSteps) * 2.0 + Math.random() * 0.3;
      lossHistory.push(Math.max(0.3, lossVal));
      updateMetric('training_loss', lossVal.toFixed(3));
      if (step % 5 === 0) {
        var evalLoss = lossVal + Math.random() * 0.1;
        updateMetric('eval_loss', evalLoss.toFixed(3));
      }
    } else if (method === 'dpo') {
      var chosenReward = 0.5 + (step / totalSteps) * 2.5 + Math.random() * 0.2;
      var rejectedReward = 0.3 + (step / totalSteps) * 0.5 + Math.random() * 0.1;
      updateMetric('rewards-chosen', chosenReward.toFixed(3));
      updateMetric('rewards-rejected', rejectedReward.toFixed(3));
      updateMetric('margin', (chosenReward - rejectedReward).toFixed(3));
      lossHistory.push(Math.max(0, 3 - (chosenReward - rejectedReward)));
    } else if (method === 'grpo') {
      var rewardMean = 2 + (step / totalSteps) * 4 + Math.random() * 0.5;
      var klDiv = Math.max(0.01, 0.5 - (step / totalSteps) * 0.3);
      updateMetric('reward_mean', rewardMean.toFixed(3));
      updateMetric('kl_divergence', klDiv.toFixed(4));
      updateMetric('score', (rewardMean / klDiv).toFixed(2));
      lossHistory.push(Math.max(0, -Math.log(rewardMean / 10)));
    }
    
    drawLossChart(info.color);
    
    if (step >= totalSteps) {
      clearInterval(trainingInterval);
      trainingInterval = null;
      finishTraining(method, info);
    }
  }, 300);
}

function updateMetric(metricId, value) {
  var el = document.getElementById('metric-' + metricId);
  if (!el) return;
  var valEl = el.querySelector('.metric-value');
  if (valEl) valEl.textContent = value;
}

function clearLossCanvas() {
  var canvas = document.getElementById('loss-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  var labels = document.getElementById('loss-labels');
  if (labels) labels.innerHTML = '<span style="color:var(--text-dim);">Loss</span>';
}

function drawLossChart(color) {
  var canvas = document.getElementById('loss-canvas');
  if (!canvas || lossHistory.length < 2) return;

  var ctx = canvas.getContext('2d');
  var width = canvas.width;
  var height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  var maxLoss = Math.max.apply(null, lossHistory);
  var minLoss = Math.min.apply(null, lossHistory);
  var range = Math.max(0.1, maxLoss - minLoss);

  // Helper to get point coords
  function getPoint(i) {
    var x = (i / Math.max(lossHistory.length - 1, 1)) * width;
    var y = height - ((lossHistory[i] - minLoss) / range) * (height - 20) - 10;
    return { x: x, y: y };
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(235,219,178,0.08)';
  ctx.lineWidth = 1;
  for (var j = 0; j < 4; j++) {
    var gridY = height * (j + 1) / 5;
    ctx.beginPath();
    ctx.moveTo(0, gridY);
    ctx.lineTo(width, gridY);
    ctx.stroke();
  }

  // Gradient fill under curve
  ctx.beginPath();
  var p0 = getPoint(0);
  ctx.moveTo(p0.x, p0.y);
  for (var i = 1; i < lossHistory.length; i++) {
    var pt = getPoint(i);
    ctx.lineTo(pt.x, pt.y);
  }
  var lastPt = getPoint(lossHistory.length - 1);
  ctx.lineTo(lastPt.x, height);
  ctx.lineTo(p0.x, height);
  ctx.closePath();

  // Parse color for gradient
  var r = parseInt(color.slice(1,3), 16) || 168;
  var g = parseInt(color.slice(3,5), 16) || 85;
  var b = parseInt(color.slice(5,7), 16) || 247;
  var grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, 'rgba(' + r + ',' + g + ',' + b + ',0.1)');
  grad.addColorStop(1, 'rgba(' + r + ',' + g + ',' + b + ',0)');
  ctx.fillStyle = grad;
  ctx.fill();

  // Glow line (wider, dimmer)
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.25;
  ctx.lineWidth = 5;
  ctx.beginPath();
  for (var i2 = 0; i2 < lossHistory.length; i2++) {
    var gp = getPoint(i2);
    if (i2 === 0) ctx.moveTo(gp.x, gp.y);
    else ctx.lineTo(gp.x, gp.y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Sharp line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (var i3 = 0; i3 < lossHistory.length; i3++) {
    var sp = getPoint(i3);
    if (i3 === 0) ctx.moveTo(sp.x, sp.y);
    else ctx.lineTo(sp.x, sp.y);
  }
  ctx.stroke();

  // Spark particle at rightmost point
  var tip = getPoint(lossHistory.length - 1);
  ctx.beginPath();
  ctx.arc(tip.x, tip.y, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.8;
  ctx.fill();
  ctx.globalAlpha = 1;

  // Outer glow on tip
  ctx.beginPath();
  ctx.arc(tip.x, tip.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.15)';
  ctx.fill();

  var labels = document.getElementById('loss-labels');
  if (labels && lossHistory.length > 5) {
    var latestLoss = lossHistory[lossHistory.length - 1];
    labels.innerHTML = '<span style="color:' + color + ';">' + latestLoss.toFixed(3) + '</span>';
  }
}

function finishTraining(method, info) {
  var stage3 = document.getElementById('llm-stage-3');
  var stage4 = document.getElementById('llm-stage-4');
  
  if (stage3) stage3.classList.add('done');
  if (stage4) stage4.classList.add('visible');
  
  var baseModel = document.getElementById('base-model')?.value || 'Qwen/Qwen2.5-0.5B';
  var epochs = document.getElementById('train-epochs')?.value || '3';
  var lr = document.getElementById('learning-rate')?.value || '1e-4';
  
  var adapterName = baseModel.split('/')[1].replace(/-/g, '_').toLowerCase() + '_' + method;
  var adapterSize = Math.round(Math.random() * 20 + 15);
  var finalLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : 0.5;
  
  var resultsContainer = document.getElementById('training-results');
  if (!resultsContainer) return;
  
  var html = '';
  
  html += '<div style="background:var(--surface); border:2px solid ' + info.color + '; border-radius:12px; padding:16px; flex:1; min-width:280px;">';
  html += '<div style="font-size:0.85rem; font-weight:700; color:' + info.color + '; margin-bottom:8px;">✓ Adapter Saved</div>';
  html += '<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.72rem; background:var(--bg); padding:10px; border-radius:6px; margin-bottom:8px;">';
  html += 'models/adapters/' + adapterName + '/';
  html += '</div>';
  html += '<div style="display:flex; gap:16px; font-size:0.68rem; color:var(--text-dim);">';
  html += '<span>Size: <strong style="color:var(--text);">' + adapterSize + 'MB</strong></span>';
  html += '<span>Steps: <strong style="color:var(--text);">' + lossHistory.length + '</strong></span>';
  html += '</div></div>';
  
  html += '<div style="background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; flex:1; min-width:280px;">';
  html += '<div style="font-size:0.85rem; font-weight:700; color:var(--cyan); margin-bottom:8px;">📊 Final Metrics</div>';
  
  if (method === 'sft') {
    var trainLoss = finalLoss.toFixed(4);
    var evalLoss = (finalLoss + 0.05).toFixed(4);
    html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">';
    html += '<div class="result-metric"><span class="label">train_loss</span><span class="value">' + trainLoss + '</span></div>';
    html += '<div class="result-metric"><span class="label">eval_loss</span><span class="value">' + evalLoss + '</span></div>';
    html += '</div>';
  } else if (method === 'dpo') {
    var margin = (2.5 + Math.random() * 0.8).toFixed(3);
    html += '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px;">';
    html += '<div class="result-metric"><span class="label">reward_margin</span><span class="value">' + margin + '</span></div>';
    html += '<div class="result-metric"><span class="label">accuracy</span><span class="value">' + (85 + Math.random() * 10).toFixed(1) + '%</span></div>';
    html += '</div>';
  } else if (method === 'grpo') {
    var score = (8 + Math.random() * 4).toFixed(2);
    html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">';
    html += '<div class="result-metric"><span class="label">final_score</span><span class="value">' + score + '</span></div>';
    html += '<div class="result-metric"><span class="label">kl_penalty</span><span class="value">' + (0.02).toFixed(4) + '</span></div>';
    html += '</div>';
  }
  
  html += '</div>';
  
  resultsContainer.innerHTML = html;

  // Show inference demo after a brief delay
  setTimeout(function() {
    var inferenceDemo = document.getElementById('inference-demo');
    if (inferenceDemo) inferenceDemo.style.display = 'block';
  }, 600);
}

var INFERENCE_RESPONSES = {
  sft: "To build an agent, you need three things: a language model, a set of tools, and an orchestration loop. First, define your tools as callable functions — web search, code execution, database queries, whatever your agent needs access to. Then write a system prompt that describes the agent's role and available tools. Finally, implement a ReAct-style loop: the model reasons about what to do, calls a tool, observes the result, and repeats until the task is done. Add memory so the agent can reference prior steps, and wrap everything in error handling so a single failed tool call doesn't crash the whole pipeline.",
  dpo: "Building an agent comes down to three core components. First, you need a language model that can follow instructions and reason about which action to take next. Second, create a tool registry — a mapping from function names to executable code, along with descriptions the model can read to decide when to use each tool. Third, build an orchestration loop that alternates between asking the model what to do and actually executing that action. The loop runs until the model signals it's finished or you hit a step limit. For production agents, you'll also want logging, retry logic, and a way to inject human oversight at critical decision points.",
  grpo: "Start with a language model and a clear definition of what tools it can use — each tool should have a name, a description, and a schema for its inputs and outputs. Then implement a planning step where the model breaks the user's request into subtasks before acting. For each subtask, the model picks a tool, you execute it, and the model reviews the result before moving on. This plan-act-observe cycle continues until the task is complete. Good agents also include a self-reflection step after each tool call to catch mistakes early, and they maintain a scratchpad of intermediate results so they don't lose context on longer tasks."
};

function runPretrainInference() {
  var btn = document.getElementById('btn-pretrain-inference');
  var output = document.getElementById('pretrain-output');
  if (!output || !btn) return;

  btn.disabled = true;
  btn.textContent = '...';
  output.style.display = 'block';
  output.innerHTML = '<span style="color:var(--gold);">Generating...</span>';

  setTimeout(function() {
    output.innerHTML = '';
    var charIdx = 0;

    function typeChar() {
      if (charIdx < PRETRAIN_RESPONSE.length) {
        output.textContent += PRETRAIN_RESPONSE[charIdx];
        charIdx++;
        setTimeout(typeChar, 18);
      } else {
        btn.disabled = false;
        btn.textContent = 'Generate';
      }
    }

    typeChar();
  }, 500);
}

function runInference() {
  var btn = document.getElementById('btn-inference');
  var output = document.getElementById('inference-output');
  if (!output || !btn) return;

  btn.disabled = true;
  btn.textContent = '...';
  output.style.display = 'block';
  output.innerHTML = '<span style="color:var(--gold);">Generating...</span>';

  var response = INFERENCE_RESPONSES[selectedMethod] || INFERENCE_RESPONSES.sft;

  setTimeout(function() {
    output.innerHTML = '';
    var charIdx = 0;

    function typeChar() {
      if (charIdx < response.length) {
        output.textContent += response[charIdx];
        charIdx++;
        setTimeout(typeChar, 25);
      } else {
        btn.disabled = false;
        btn.textContent = 'Generate';
      }
    }

    typeChar();
  }, 500);
}

function resetLLMDemo() {
  if (trainingInterval) {
    clearInterval(trainingInterval);
    trainingInterval = null;
  }
  
  selectedMethod = null;
  lossHistory = [];
  
  document.querySelectorAll('.method-card').forEach(function(card) {
    card.classList.remove('selected');
  });
  
  var methodSelect = document.getElementById('training-method');
  if (methodSelect) methodSelect.value = '';
  
  var explanation = document.getElementById('method-explanation');
  if (explanation) explanation.style.display = 'none';
  
  ['llm-stage-1', 'llm-stage-2', 'llm-stage-3', 'llm-stage-4'].forEach(function(id) {
    var stage = document.getElementById(id);
    if (!stage) return;
    stage.classList.remove('done');
    if (id === 'llm-stage-1') {
      stage.classList.add('visible');
    } else {
      stage.classList.remove('visible');
    }
  });
  
  var progressBar = document.getElementById('train-progress-bar');
  if (progressBar) progressBar.style.width = '0%';
  
  var progressPct = document.getElementById('train-progress-pct');
  if (progressPct) progressPct.textContent = '0%';
  
  clearLossCanvas();
  
  var metricsContainer = document.getElementById('training-metrics');
  if (metricsContainer) metricsContainer.innerHTML = '';
  
  var resultsContainer = document.getElementById('training-results');
  if (resultsContainer) resultsContainer.innerHTML = '';

  var inferenceDemo = document.getElementById('inference-demo');
  if (inferenceDemo) inferenceDemo.style.display = 'none';
  var inferenceOutput = document.getElementById('inference-output');
  if (inferenceOutput) { inferenceOutput.style.display = 'none'; inferenceOutput.innerHTML = ''; }
  var pretrainOutput = document.getElementById('pretrain-output');
  if (pretrainOutput) { pretrainOutput.style.display = 'none'; pretrainOutput.innerHTML = ''; }
  var pretrainBtn = document.getElementById('btn-pretrain-inference');
  if (pretrainBtn) { pretrainBtn.disabled = false; pretrainBtn.textContent = 'Generate'; }
}

registerAnim(8,
  function enter() {
    resetLLMDemo();
  },
  function leave() {
    if (trainingInterval) {
      clearInterval(trainingInterval);
      trainingInterval = null;
    }
  }
);