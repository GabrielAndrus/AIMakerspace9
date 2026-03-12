/* ══════════════════════════════════════════════════════════════
   SLIDE 9: TRACE WATERFALL + CHART.JS RADAR (slide index 8)
   ══════════════════════════════════════════════════════════════ */

let radarChart = null;
let waterfallAnimated = false;

function animateWaterfall() {
  if (waterfallAnimated) return;
  waterfallAnimated = true;
  const bars = document.querySelectorAll('#waterfall .waterfall-bar');
  bars.forEach((bar, i) => {
    setTimeout(() => {
      bar.style.width = bar.dataset.width;
    }, i * 200);
  });
}

function buildRadarChart(data) {
  const ctx = document.getElementById('ragas-radar');
  if (!ctx) return;

  if (radarChart) { radarChart.destroy(); }

  const defaults = data || {
    dense:  [0.82, 0.75, 0.88],
    sparse: [0.65, 0.60, 0.72],
    hybrid: [0.90, 0.85, 0.91],
  };

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Faithfulness', 'Context Precision', 'Context Recall'],
      datasets: [
        {
          label: 'Dense',
          data: defaults.dense,
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(34,211,238,0.12)',
          borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#22d3ee',
        },
        {
          label: 'Sparse (BM25)',
          data: defaults.sparse,
          borderColor: '#818cf8',
          backgroundColor: 'rgba(129,140,248,0.10)',
          borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#818cf8',
        },
        {
          label: 'Hybrid (RRF)',
          data: defaults.hybrid,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.10)',
          borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#f59e0b',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 1200, easing: 'easeOutQuart' },
      scales: {
        r: {
          beginAtZero: true, max: 1, min: 0,
          ticks: { stepSize: 0.25, color: '#64748b', backdropColor: 'transparent', font: { size: 10 } },
          grid: { color: '#2e3648' },
          angleLines: { color: '#2e3648' },
          pointLabels: { color: '#e2e8f0', font: { size: 12, weight: '600' } },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

async function runLiveRagas() {
  const result = await gradioCall('run_ragas_evaluation', ['dense'], null);
  if (result) {
    // Try to parse live results
    try {
      buildRadarChart(result);
    } catch { buildRadarChart(null); }
  } else {
    // Fallback: re-render with defaults
    buildRadarChart(null);
  }
}

registerAnim(8,
  function enter() {
    animateWaterfall();
    // Small delay so Chart.js renders after the slide is visible
    setTimeout(() => buildRadarChart(null), 300);
  },
  function leave() {
    // Chart.js is static once rendered, no need to destroy on leave
  }
);
