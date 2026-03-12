
bar.style.width = (1 / TOTAL * 100) + '%';

// Status check
(async () => {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  try {
    const r = await fetch(GRADIO + '/info', { signal: AbortSignal.timeout(3000) });
    if (r.ok) { dot.classList.add('on'); txt.textContent = 'Platform is live at :7860'; }
    else { dot.classList.add('off'); txt.textContent = 'Gradio returned ' + r.status; }
  } catch { dot.classList.add('off'); txt.textContent = 'Not running \u2014 start with docker compose up -d'; }
})();

// Fire enter for initial slide
if (slideAnims[0] && slideAnims[0].enter) {
  slideAnims[0].enter(0);
}
