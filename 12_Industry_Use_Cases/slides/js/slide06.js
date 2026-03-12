/* ══════════════════════════════════════════════════════════════
   SLIDE 6: DECISION TREE (slide index 5)
   ══════════════════════════════════════════════════════════════ */

function buildDecisionTree() {
  const svg = document.getElementById('dtree-svg');
  if (svg.childNodes.length > 0) return;

  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  // Tree structure
  // Root -> diamond "prompt+chosen+rejected?" -> Yes: DPO leaf, No -> diamond "messages with assistant?" -> Yes: SFT leaf, No -> diamond "prompt+ground_truth?" -> Yes: GRPO leaf

  // Root node
  const root = el('rect', { x: '330', y: '15', width: '140', height: '36', rx: '8', fill: 'var(--surface)', stroke: '#e2e8f0', 'stroke-width': '2' });
  svg.appendChild(root);
  const rootText = el('text', { x: '400', y: '38', 'text-anchor': 'middle', fill: '#e2e8f0', 'font-size': '11', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
  rootText.textContent = 'Upload Training Data';
  svg.appendChild(rootText);

  // Diamond 1: prompt+chosen+rejected?
  const d1 = el('polygon', { points: '400,80 480,115 400,150 320,115', fill: 'var(--surface2)', stroke: '#10b981', 'stroke-width': '1.5', class: 'dtree-node', id: 'dtree-d1' });
  svg.appendChild(d1);
  const d1Text = el('text', { x: '400', y: '110', 'text-anchor': 'middle', fill: '#10b981', 'font-size': '10', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
  d1Text.textContent = 'Has preference pairs?';
  const d1Sub = el('text', { x: '400', y: '122', 'text-anchor': 'middle', fill: '#64748b', 'font-size': '8', 'font-family': 'Inter, sans-serif' });
  d1Sub.textContent = '(good vs bad examples)';
  svg.appendChild(d1Sub);
  svg.appendChild(d1Text);

  // Path root -> d1
  const p1 = el('line', { x1: '400', y1: '51', x2: '400', y2: '80', stroke: '#2e3648', 'stroke-width': '2', class: 'dtree-path', id: 'dtree-p1' });
  svg.appendChild(p1);

  // DPO leaf (right of d1)
  const dpoLeaf = el('rect', { x: '530', y: '98', width: '70', height: '34', rx: '6', fill: 'rgba(16,185,129,0.15)', stroke: '#10b981', 'stroke-width': '2', id: 'dtree-leaf-dpo' });
  svg.appendChild(dpoLeaf);
  const dpoText = el('text', { x: '565', y: '116', 'text-anchor': 'middle', fill: '#10b981', 'font-size': '13', 'font-weight': '800', 'font-family': 'Inter, sans-serif' });
  dpoText.textContent = 'DPO';
  svg.appendChild(dpoText);
  const dpoDesc = el('text', { x: '565', y: '145', 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '8', 'font-family': 'Inter, sans-serif' });
  dpoDesc.textContent = 'Learn from preferences';
  svg.appendChild(dpoDesc);
  const p2 = el('line', { x1: '480', y1: '115', x2: '530', y2: '115', stroke: '#10b981', 'stroke-width': '2', class: 'dtree-path', id: 'dtree-p-dpo' });
  svg.appendChild(p2);
  const yesLabel1 = el('text', { x: '500', y: '108', fill: '#10b981', 'font-size': '9', 'font-weight': '600' });
  yesLabel1.textContent = 'Yes';
  svg.appendChild(yesLabel1);

  // No path down from d1
  const p3 = el('line', { x1: '400', y1: '150', x2: '400', y2: '185', stroke: '#2e3648', 'stroke-width': '2', class: 'dtree-path', id: 'dtree-p3' });
  svg.appendChild(p3);
  const noLabel1 = el('text', { x: '410', y: '172', fill: '#ef4444', 'font-size': '9', 'font-weight': '600' });
  noLabel1.textContent = 'No';
  svg.appendChild(noLabel1);

  // Diamond 2: messages with assistant?
  const d2 = el('polygon', { points: '400,185 470,215 400,245 330,215', fill: 'var(--surface2)', stroke: '#818cf8', 'stroke-width': '1.5', class: 'dtree-node', id: 'dtree-d2' });
  svg.appendChild(d2);
  const d2Text = el('text', { x: '400', y: '213', 'text-anchor': 'middle', fill: '#818cf8', 'font-size': '10', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
  d2Text.textContent = 'Has chat conversations?';
  const d2Sub = el('text', { x: '400', y: '224', 'text-anchor': 'middle', fill: '#64748b', 'font-size': '8', 'font-family': 'Inter, sans-serif' });
  d2Sub.textContent = '(user/assistant messages)';
  svg.appendChild(d2Sub);
  svg.appendChild(d2Text);

  // SFT leaf (left of d2)
  const sftLeaf = el('rect', { x: '200', y: '198', width: '70', height: '34', rx: '6', fill: 'rgba(129,140,248,0.12)', stroke: '#818cf8', 'stroke-width': '2', id: 'dtree-leaf-sft' });
  svg.appendChild(sftLeaf);
  const sftText = el('text', { x: '235', y: '216', 'text-anchor': 'middle', fill: '#818cf8', 'font-size': '13', 'font-weight': '800', 'font-family': 'Inter, sans-serif' });
  sftText.textContent = 'SFT';
  svg.appendChild(sftText);
  const sftDesc = el('text', { x: '235', y: '245', 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '8', 'font-family': 'Inter, sans-serif' });
  sftDesc.textContent = 'Learn from examples';
  svg.appendChild(sftDesc);
  const p4 = el('line', { x1: '330', y1: '215', x2: '270', y2: '215', stroke: '#818cf8', 'stroke-width': '2', class: 'dtree-path', id: 'dtree-p-sft' });
  svg.appendChild(p4);
  const yesLabel2 = el('text', { x: '293', y: '208', fill: '#818cf8', 'font-size': '9', 'font-weight': '600' });
  yesLabel2.textContent = 'Yes';
  svg.appendChild(yesLabel2);

  // No path down from d2
  const p5 = el('line', { x1: '400', y1: '245', x2: '400', y2: '275', stroke: '#2e3648', 'stroke-width': '2', class: 'dtree-path', id: 'dtree-p5' });
  svg.appendChild(p5);
  const noLabel2 = el('text', { x: '410', y: '264', fill: '#ef4444', 'font-size': '9', 'font-weight': '600' });
  noLabel2.textContent = 'No';
  svg.appendChild(noLabel2);

  // Diamond 3: prompt+ground_truth?
  const d3 = el('polygon', { points: '400,275 470,305 400,335 330,305', fill: 'var(--surface2)', stroke: '#f59e0b', 'stroke-width': '1.5', class: 'dtree-node', id: 'dtree-d3' });
  svg.appendChild(d3);
  const d3Text = el('text', { x: '400', y: '303', 'text-anchor': 'middle', fill: '#f59e0b', 'font-size': '10', 'font-weight': '700', 'font-family': 'Inter, sans-serif' });
  d3Text.textContent = 'Has correct answers?';
  const d3Sub = el('text', { x: '400', y: '314', 'text-anchor': 'middle', fill: '#64748b', 'font-size': '8', 'font-family': 'Inter, sans-serif' });
  d3Sub.textContent = '(questions + ground truth)';
  svg.appendChild(d3Sub);
  svg.appendChild(d3Text);

  // GRPO leaf (right of d3)
  const grpoLeaf = el('rect', { x: '530', y: '288', width: '70', height: '34', rx: '6', fill: 'rgba(245,158,11,0.15)', stroke: '#f59e0b', 'stroke-width': '2', id: 'dtree-leaf-grpo' });
  svg.appendChild(grpoLeaf);
  const grpoText = el('text', { x: '565', y: '306', 'text-anchor': 'middle', fill: '#f59e0b', 'font-size': '13', 'font-weight': '800', 'font-family': 'Inter, sans-serif' });
  grpoText.textContent = 'GRPO';
  svg.appendChild(grpoText);
  const grpoDesc = el('text', { x: '565', y: '335', 'text-anchor': 'middle', fill: '#94a3b8', 'font-size': '8', 'font-family': 'Inter, sans-serif' });
  grpoDesc.textContent = 'Learn from rewards';
  svg.appendChild(grpoDesc);
  const p6 = el('line', { x1: '470', y1: '305', x2: '530', y2: '305', stroke: '#f59e0b', 'stroke-width': '2', class: 'dtree-path', id: 'dtree-p-grpo' });
  svg.appendChild(p6);
  const yesLabel3 = el('text', { x: '494', y: '298', fill: '#f59e0b', 'font-size': '9', 'font-weight': '600' });
  yesLabel3.textContent = 'Yes';
  svg.appendChild(yesLabel3);
}

let dtreeDrawn = false;
function drawTreePaths() {
  if (dtreeDrawn) return;
  dtreeDrawn = true;
  const paths = ['dtree-p1', 'dtree-p-dpo', 'dtree-p3', 'dtree-p-sft', 'dtree-p5', 'dtree-p-grpo'];
  paths.forEach((id, i) => {
    setTimeout(() => {
      document.getElementById(id)?.classList.add('drawn');
    }, i * 400);
  });
}

const DTREE_FALLBACK = {
  sft: { recommended_method: 'SFT', reasoning: 'Dataset contains messages with assistant role - conversational format detected.', issues: [], suggestions: ['Add system prompts for richer context'] },
  dpo: { recommended_method: 'DPO', reasoning: 'Dataset contains prompt/chosen/rejected preference pairs.', issues: [], suggestions: ['Ensure chosen responses are measurably better than rejected'] },
  grpo: { recommended_method: 'GRPO', reasoning: 'Dataset contains prompt + ground_truth fields suitable for reward modeling.', issues: [], suggestions: ['Consider adding difficulty labels for curriculum learning'] },
};

async function triggerDTree(method) {
  // Highlight buttons
  document.querySelectorAll('.dtree-btn').forEach(b => b.className = 'dtree-btn');
  const activeBtn = event.target;
  activeBtn.classList.add(`active-${method}`);

  // Highlight matching path, dim others
  const pathMap = {
    dpo: ['dtree-p1', 'dtree-p-dpo'],
    sft: ['dtree-p1', 'dtree-p3', 'dtree-p-sft'],
    grpo: ['dtree-p1', 'dtree-p3', 'dtree-p5', 'dtree-p-grpo'],
  };
  const allPaths = ['dtree-p1', 'dtree-p-dpo', 'dtree-p3', 'dtree-p-sft', 'dtree-p5', 'dtree-p-grpo'];
  const activePaths = pathMap[method];
  allPaths.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('highlight', 'dimmed');
    if (activePaths.includes(id)) {
      el.classList.add('highlight');
    } else {
      el.classList.add('dimmed');
    }
  });

  // Show output
  const out = document.getElementById('dtree-output');
  out.classList.add('visible');
  out.textContent = '';

  // Try live API
  const result = await gradioCall('handle_llm_dataset_upload', [null], null);
  const data = result || DTREE_FALLBACK[method];
  // Typewriter the JSON
  const json = JSON.stringify(data, null, 2);
  let i = 0;
  function typeChar() {
    if (i < json.length) {
      out.textContent += json[i];
      i++;
      setTimeout(typeChar, 15);
    }
  }
  typeChar();
}

registerAnim(5,
  function enter() { buildDecisionTree(); drawTreePaths(); },
  null
);
