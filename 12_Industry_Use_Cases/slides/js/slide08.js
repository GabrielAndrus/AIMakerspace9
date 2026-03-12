/* ══════════════════════════════════════════════════════════════
   SLIDE 8: SPLIT-SCREEN ANIMATION (slide index 7)
   ══════════════════════════════════════════════════════════════ */

let splitPlayed = false;
function playSplitScreen() {
  if (splitPlayed) return;
  splitPlayed = true;

  const steps = ['mock-1', 'mock-2', 'mock-3', 'mock-4'];
  const counter = document.getElementById('click-counter');
  let clickCount = 0;

  // Left side: show steps
  steps.forEach((id, i) => {
    setTimeout(() => {
      document.getElementById(id)?.classList.add('visible');
      clickCount++;
      if (counter) counter.textContent = String(clickCount);
    }, 400 + i * 700);
  });

  // Right side: gantt bars (start simultaneously)
  const ganttBars = document.querySelectorAll('#split-screen .gantt-fill');
  ganttBars.forEach((bar, i) => {
    setTimeout(() => {
      bar.style.width = bar.dataset.width;
    }, 600 + i * 300);
  });

  // Stats at end
  setTimeout(() => {
    document.getElementById('split-stats')?.classList.add('visible');
  }, 600 + ganttBars.length * 300 + 500);
}

registerAnim(7,
  function enter() { playSplitScreen(); },
  null
);
