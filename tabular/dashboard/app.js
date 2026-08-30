/**
 * tabular/dashboard/app.js
 * Frontend state management, API synchronization & multi-model comparison tables.
 */

let pipelineData = { pipeline_stages: [] };
let configData = {};
let resultsData = { tasks_status: [], metrics: [] };
let pollingInterval = null;
let isCurrentlyRunning = false;

// Filter state for comparison table
let activeTcFilter = 'all';
let activeModelFilter = 'all';
let activeArchFilter = 'all';

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initPresets();
  initActions();
  initFilterPills();
  loadPipeline();
  loadConfig();
  loadResults();
  startStatusPolling();
});

// ----------------------------------------------------------------------
// 1. Tab Navigation
// ----------------------------------------------------------------------
function initTabs() {
  const navItems = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const pageTitle = document.getElementById('pageTitle');
  const pageSubtitle = document.getElementById('pageSubtitle');

  const titles = {
    'pipeline-tab': {
      title: 'Pipeline Orchestrator',
      subtitle: 'Configure and trigger few-shot meta-learning workflows in chronological order'
    },
    'results-tab': {
      title: 'Results & Model Comparison',
      subtitle: 'Side-by-side performance breakdown across models, architectures, and test cases'
    },
    'config-tab': {
      title: 'Hyperparameters & Dataset Config',
      subtitle: 'Adjust training splits, support/query steps, and model optimization parameters'
    },
    'terminal-tab': {
      title: 'Raw Subprocess Console',
      subtitle: 'Raw stdout/stderr stream from background processes'
    }
  };

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetTab = item.getAttribute('data-tab');

      navItems.forEach(n => n.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      item.classList.add('active');
      const pane = document.getElementById(targetTab);
      if (pane) pane.classList.add('active');

      if (titles[targetTab]) {
        pageTitle.textContent = titles[targetTab].title;
        pageSubtitle.textContent = titles[targetTab].subtitle;
      }

      if (targetTab === 'results-tab') {
        loadResults();
      }
    });
  });
}

// ----------------------------------------------------------------------
// 2. Pipeline Loading & Rendering
// ----------------------------------------------------------------------
async function loadPipeline() {
  try {
    const res = await fetch('/api/pipeline');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    pipelineData = await res.json();
    renderPipelineStages();
  } catch (err) {
    showToast(`Failed to load pipeline: ${err.message}`, 'error');
  }
}

function renderPipelineStages() {
  const container = document.getElementById('stagesContainer');
  container.innerHTML = '';

  if (!pipelineData.pipeline_stages || pipelineData.pipeline_stages.length === 0) {
    container.innerHTML = '<div class="loading-spinner">No pipeline stages found.</div>';
    return;
  }

  pipelineData.pipeline_stages.forEach((stage, stageIdx) => {
    const stageEl = document.createElement('div');
    stageEl.className = 'stage-section';

    stageEl.innerHTML = `
      <div class="stage-header">
        <div class="stage-title">
          <span class="stage-num">${stageIdx + 1}</span>
          <div>
            <h2>${stage.stage_name}</h2>
            <div class="stage-desc">${stage.description || ''}</div>
          </div>
        </div>
      </div>
      <div class="scripts-grid" id="stageGrid_${stageIdx}"></div>
    `;

    container.appendChild(stageEl);
    const grid = stageEl.querySelector(`#stageGrid_${stageIdx}`);

    stage.scripts.forEach((script, scriptIdx) => {
      const card = document.createElement('div');
      card.className = 'script-card';

      // Badges
      let badgesHtml = '';
      if (script.architecture) {
        badgesHtml += `<span class="badge badge-arch">${script.architecture}</span>`;
      }
      if (script.model) {
        const modelLower = script.model.toLowerCase();
        badgesHtml += `<span class="badge badge-${modelLower}">${script.model}</span>`;
      }

      card.innerHTML = `
        <div class="script-info">
          <div class="script-name">${script.name}</div>
          <div class="script-path">${script.path}</div>
          <div class="script-meta">${badgesHtml}</div>
        </div>
        <label class="switch">
          <input type="checkbox" data-stage="${stageIdx}" data-script="${scriptIdx}" ${script.enabled ? 'checked' : ''}>
          <span class="slider"></span>
        </label>
      `;

      grid.appendChild(card);

      const toggle = card.querySelector('input[type="checkbox"]');
      toggle.addEventListener('change', (e) => {
        pipelineData.pipeline_stages[stageIdx].scripts[scriptIdx].enabled = e.target.checked;
        savePipeline(false);
      });
    });
  });
}

async function savePipeline(showNotification = true) {
  try {
    const res = await fetch('/api/pipeline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pipelineData)
    });
    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }
    const data = await res.json();
    if (showNotification) {
      showToast('Pipeline order and toggles saved successfully!', 'success');
    }
  } catch (err) {
    const isOffline = err.message.includes('Failed to fetch') || err.message.includes('NetworkError');
    const msg = isOffline
      ? 'Backend disconnected. Please ensure "python tabular/dashboard/app.py" is running.'
      : `Failed to save pipeline: ${err.message}`;
    showToast(msg, 'error');
  }
}

// ----------------------------------------------------------------------
// 3. Quick Presets
// ----------------------------------------------------------------------
function initPresets() {
  document.querySelectorAll('.btn-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const preset = btn.getAttribute('data-preset');
      applyPreset(preset);
    });
  });

  document.getElementById('btnSavePipeline').addEventListener('click', () => {
    savePipeline(true);
  });
}

function applyPreset(preset) {
  if (!pipelineData.pipeline_stages) return;

  pipelineData.pipeline_stages.forEach(stage => {
    stage.scripts.forEach(script => {
      const model = (script.model || '').toLowerCase();
      const arch = (script.architecture || '').toLowerCase();
      const stageId = stage.stage_id.toLowerCase();

      switch (preset) {
        case 'all':
          script.enabled = true;
          break;
        case 'none':
          script.enabled = false;
          break;
        case 'maml':
          script.enabled = (stageId.includes('preprocessing') || model === 'maml');
          break;
        case 'platipus':
          script.enabled = (stageId.includes('preprocessing') || model === 'platipus');
          break;
        case 'vampire':
          script.enabled = (stageId.includes('preprocessing') || model === 'vampire');
          break;
        case 'abml':
          script.enabled = (stageId.includes('preprocessing') || model === 'abml');
          break;
        case 'bmaml':
          script.enabled = (stageId.includes('preprocessing') || model === 'bmaml');
          break;
        case 'fcnet':
          script.enabled = (stageId.includes('preprocessing') || arch === 'fcnet');
          break;
        case 'lr':
          script.enabled = (stageId.includes('preprocessing') || arch.includes('logistic'));
          break;
        case 'eval_only':
          script.enabled = (stageId.includes('eval') || stageId.includes('adaptation'));
          break;
      }
    });
  });

  renderPipelineStages();
  savePipeline(false);
  showToast(`Applied preset: ${preset.toUpperCase()}`, 'success');
}

// ----------------------------------------------------------------------
// 4. Results & Comparison Table Rendering
// ----------------------------------------------------------------------
async function loadResults() {
  try {
    const res = await fetch('/api/results');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    resultsData = await res.json();
    renderTasksStatus(resultsData.tasks_status || []);
    renderComparisonTable(resultsData.metrics || []);
  } catch (err) {
    // Keep quiet on initial load if file doesn't exist yet
  }
}

function renderTasksStatus(tasks) {
  const grid = document.getElementById('tasksStatusGrid');
  const badge = document.getElementById('taskCountBadge');
  grid.innerHTML = '';

  if (!tasks || tasks.length === 0) {
    grid.innerHTML = '<div class="empty-state">No pipeline tasks executed yet. Click "Run Pipeline" to begin.</div>';
    badge.textContent = '0 Tasks';
    return;
  }

  badge.textContent = `${tasks.length} Tasks`;

  tasks.forEach(t => {
    const card = document.createElement('div');
    const isSuccess = (t.status === 'SUCCESS');
    card.className = `task-card ${isSuccess ? 'success' : 'failed'}`;

    card.innerHTML = `
      <div class="task-card-header">
        <span class="task-card-title">${isSuccess ? '✅' : '❌'} ${t.name}</span>
        <span class="task-card-duration">${t.duration}</span>
      </div>
      <div class="task-card-msg">${t.message || t.stage}</div>
    `;
    grid.appendChild(card);
  });
}

function initFilterPills() {
  document.getElementById('btnRefreshResults').addEventListener('click', loadResults);
  document.getElementById('btnExportCsv').addEventListener('click', exportResultsCsv);
  document.getElementById('btnExportJson').addEventListener('click', exportResultsJson);
  document.getElementById('btnDeleteAll').addEventListener('click', deleteAllResults);
}

async function deleteAllResults() {
  if (!confirm('Are you sure you want to delete all recorded metrics and task execution results?')) {
    return;
  }

  try {
    const res = await fetch('/api/results/clear', { method: 'POST' });
    const data = await res.json();
    resultsData = { tasks_status: [], metrics: [] };
    renderTasksStatus([]);
    renderComparisonTable([]);
    showToast('All results and metrics deleted successfully!', 'success');
  } catch (err) {
    showToast(`Failed to delete results: ${err.message}`, 'error');
  }
}

function updateTestCaseFilterPills(metrics) {
  const container = document.getElementById('testCaseFilterPills');
  if (!container) return;
  const uniqueTcs = sortedUnique(metrics.map(m => m.test_case));

  let html = `<button class="pill ${activeTcFilter === 'all' ? 'active' : ''}" data-filter-tc="all">All Test Cases</button>`;
  uniqueTcs.forEach(tc => {
    const isActive = (activeTcFilter === tc);
    html += `<button class="pill ${isActive ? 'active' : ''}" data-filter-tc="${tc}">${tc}</button>`;
  });
  container.innerHTML = html;

  container.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      container.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      activeTcFilter = btn.getAttribute('data-filter-tc');
      renderComparisonTable(resultsData.metrics || []);
    });
  });
}

function updateModelFilterPills(metrics) {
  const container = document.getElementById('modelFilterPills');
  if (!container) return;
  
  // Standard techniques plus any discovered from metrics
  const defaultModels = ['MAML', 'PLATIPUS', 'VAMPIRE', 'ABML', 'BMAML'];
  const metricsModels = sortedUnique(metrics.map(m => m.model));
  const allModels = Array.from(new Set([...defaultModels, ...metricsModels]));

  let html = `<button class="pill ${activeModelFilter === 'all' ? 'active' : ''}" data-filter-model="all">All Techniques</button>`;
  allModels.forEach(m => {
    const isActive = (activeModelFilter.toLowerCase() === m.toLowerCase());
    html += `<button class="pill ${isActive ? 'active' : ''}" data-filter-model="${m}">${m}</button>`;
  });
  container.innerHTML = html;

  container.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      container.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      activeModelFilter = btn.getAttribute('data-filter-model');
      renderComparisonTable(resultsData.metrics || []);
    });
  });
}

function updateArchFilterPills(metrics) {
  const container = document.getElementById('archFilterPills');
  if (!container) return;

  const defaultArchs = [
    { id: 'FcNet', label: 'FcNet' },
    { id: 'LogisticRegression', label: 'Logistic Regression' }
  ];
  
  let html = `<button class="pill ${activeArchFilter === 'all' ? 'active' : ''}" data-filter-arch="all">All Architectures</button>`;
  defaultArchs.forEach(a => {
    const isActive = (activeArchFilter.toLowerCase() === a.id.toLowerCase());
    html += `<button class="pill ${isActive ? 'active' : ''}" data-filter-arch="${a.id}">${a.label}</button>`;
  });
  container.innerHTML = html;

  container.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      container.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      activeArchFilter = btn.getAttribute('data-filter-arch');
      renderComparisonTable(resultsData.metrics || []);
    });
  });
}

function sortedUnique(arr) {
  return Array.from(new Set(arr.filter(Boolean))).sort();
}

function renderComparisonTable(metrics) {
  updateTestCaseFilterPills(metrics);
  updateModelFilterPills(metrics);
  updateArchFilterPills(metrics);

  const tbody = document.getElementById('metricsTableBody');
  tbody.innerHTML = '';

  if (!metrics || metrics.length === 0) {
    tbody.innerHTML = '<tr><td colspan="15" class="empty-table-msg">No evaluation metrics recorded yet.</td></tr>';
    return;
  }

  // Group metrics by (test_case, model, architecture)
  const grouped = {};
  metrics.forEach(m => {
    // Apply Test Case filter
    if (activeTcFilter !== 'all' && m.test_case !== activeTcFilter) {
      return;
    }

    const key = `${m.test_case}__${m.model}__${m.architecture}`;
    if (!grouped[key]) {
      grouped[key] = {
        test_case: m.test_case,
        model: m.model,
        architecture: m.architecture,
        adapted: null,
        cold: null
      };
    }

    const mode = (m.mode || '').toLowerCase();
    if (mode.includes('fast') || mode.includes('adapt')) {
      grouped[key].adapted = m;
    } else {
      grouped[key].cold = m;
    }
  });

  const rowKeys = Object.keys(grouped).sort();
  if (rowKeys.length === 0) {
    tbody.innerHTML = '<tr><td colspan="15" class="empty-table-msg">No metrics match the current filter selection.</td></tr>';
    return;
  }

  // Filter by Meta Technique (Model) and Architecture
  const rows = [];
  rowKeys.forEach(k => {
    const item = grouped[k];
    if (activeModelFilter !== 'all' && (item.model || '').toLowerCase() !== activeModelFilter.toLowerCase()) {
      return;
    }
    if (activeArchFilter !== 'all' && (item.architecture || '').toLowerCase() !== activeArchFilter.toLowerCase()) {
      return;
    }
    rows.push(item);
  });

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="15" class="empty-table-msg">No metrics match the current technique / architecture filter selection.</td></tr>';
    return;
  }

  rows.forEach(item => {
    const tr = document.createElement('tr');

    const modelBadge = `<span class="badge badge-${(item.model || '').toLowerCase()}">${item.model}</span>`;
    const archBadge = `<span class="badge badge-arch">${item.architecture}</span>`;

    const ad = item.adapted;
    const cd = item.cold;

    // Row-level comparison classes
    let adAccClass = 'metric-num';
    let cdAccClass = 'metric-num';
    let adAccBadge = '';
    let cdAccBadge = '';

    let adMacroClass = 'metric-num';
    let cdMacroClass = 'metric-num';

    let adC1F1Class = 'metric-num';
    let cdC1F1Class = 'metric-num';

    if (ad && cd) {
      // 1. Accuracy comparison
      if (ad.accuracy > cd.accuracy) {
        const diff = (ad.accuracy - cd.accuracy).toFixed(2);
        adAccClass = 'metric-win';
        cdAccClass = 'metric-loss';
        adAccBadge = `<span title="+${diff}% vs Cold-Start">▲</span>`;
        cdAccBadge = `<span title="-${diff}% vs Adapted">▼</span>`;
      } else if (ad.accuracy < cd.accuracy) {
        const diff = (cd.accuracy - ad.accuracy).toFixed(2);
        adAccClass = 'metric-loss';
        cdAccClass = 'metric-win';
        adAccBadge = `<span title="-${diff}% vs Cold-Start">▼</span>`;
        cdAccBadge = `<span title="+${diff}% vs Adapted">▲</span>`;
      } else {
        adAccClass = 'metric-tie';
        cdAccClass = 'metric-tie';
      }

      // 2. Macro F1 comparison
      if (ad.macro_f1 > cd.macro_f1) {
        adMacroClass = 'metric-win';
        cdMacroClass = 'metric-loss';
      } else if (ad.macro_f1 < cd.macro_f1) {
        adMacroClass = 'metric-loss';
        cdMacroClass = 'metric-win';
      } else {
        adMacroClass = 'metric-tie';
        cdMacroClass = 'metric-tie';
      }

      // 3. Class 1 (Smoking) F1 comparison
      const adC1F1 = (ad.class_1 && ad.class_1.f1 !== undefined) ? ad.class_1.f1 : 0;
      const cdC1F1 = (cd.class_1 && cd.class_1.f1 !== undefined) ? cd.class_1.f1 : 0;
      if (adC1F1 > cdC1F1) {
        adC1F1Class = 'metric-win';
        cdC1F1Class = 'metric-loss';
      } else if (adC1F1 < cdC1F1) {
        adC1F1Class = 'metric-loss';
        cdC1F1Class = 'metric-win';
      } else {
        adC1F1Class = 'metric-tie';
        cdC1F1Class = 'metric-tie';
      }
    } else if (ad) {
      adAccClass = 'metric-tie';
      adMacroClass = 'metric-tie';
      adC1F1Class = 'metric-tie';
    } else if (cd) {
      cdAccClass = 'metric-tie';
      cdMacroClass = 'metric-tie';
      cdC1F1Class = 'metric-tie';
    }

    // 1. Adapted columns helper
    let adaptedHtml = '';
    if (ad) {
      const c0 = ad.class_0 || {};
      const c1 = ad.class_1 || {};
      const c0Pr = `${c0.precision !== undefined ? c0.precision.toFixed(1) : '0.0'}% / ${c0.recall !== undefined ? c0.recall.toFixed(1) : '0.0'}%`;
      const c1Pr = `${c1.precision !== undefined ? c1.precision.toFixed(1) : '0.0'}% / ${c1.recall !== undefined ? c1.recall.toFixed(1) : '0.0'}%`;
      const cm = ad.confusion_matrix || [[0, 0], [0, 0]];
      const cmStr = `[${cm[0][0]}, ${cm[0][1]} / ${cm[1][0]}, ${cm[1][1]}]`;

      adaptedHtml = `
        <td class="adapted-col"><span class="${adAccClass}">${ad.accuracy.toFixed(2)}% ${adAccBadge}</span></td>
        <td class="adapted-col"><span class="${adMacroClass}">${ad.macro_f1.toFixed(2)}%</span></td>
        <td class="adapted-col"><span class="metric-num-sub">${c0Pr}</span></td>
        <td class="adapted-col"><span class="metric-num-sub">${c1Pr}</span></td>
        <td class="adapted-col"><span class="${adC1F1Class}">${c1.f1 !== undefined ? c1.f1.toFixed(2) : '0.00'}%</span></td>
        <td class="adapted-col"><span class="cm-box">${cmStr}</span></td>
      `;
    } else {
      adaptedHtml = `
        <td class="adapted-col empty-col">-</td>
        <td class="adapted-col empty-col">-</td>
        <td class="adapted-col empty-col">-</td>
        <td class="adapted-col empty-col">-</td>
        <td class="adapted-col empty-col">-</td>
        <td class="adapted-col empty-col">-</td>
      `;
    }

    // 2. Cold-Start (Cooldown) columns helper
    let coldHtml = '';
    if (cd) {
      const c0 = cd.class_0 || {};
      const c1 = cd.class_1 || {};
      const c0Pr = `${c0.precision !== undefined ? c0.precision.toFixed(1) : '0.0'}% / ${c0.recall !== undefined ? c0.recall.toFixed(1) : '0.0'}%`;
      const c1Pr = `${c1.precision !== undefined ? c1.precision.toFixed(1) : '0.0'}% / ${c1.recall !== undefined ? c1.recall.toFixed(1) : '0.0'}%`;
      const cm = cd.confusion_matrix || [[0, 0], [0, 0]];
      const cmStr = `[${cm[0][0]}, ${cm[0][1]} / ${cm[1][0]}, ${cm[1][1]}]`;

      coldHtml = `
        <td class="cold-col"><span class="${cdAccClass}">${cd.accuracy.toFixed(2)}% ${cdAccBadge}</span></td>
        <td class="cold-col"><span class="${cdMacroClass}">${cd.macro_f1.toFixed(2)}%</span></td>
        <td class="cold-col"><span class="metric-num-sub">${c0Pr}</span></td>
        <td class="cold-col"><span class="metric-num-sub">${c1Pr}</span></td>
        <td class="cold-col"><span class="${cdC1F1Class}">${c1.f1 !== undefined ? c1.f1.toFixed(2) : '0.00'}%</span></td>
        <td class="cold-col"><span class="cm-box">${cmStr}</span></td>
      `;
    } else {
      coldHtml = `
        <td class="cold-col empty-col">-</td>
        <td class="cold-col empty-col">-</td>
        <td class="cold-col empty-col">-</td>
        <td class="cold-col empty-col">-</td>
        <td class="cold-col empty-col">-</td>
        <td class="cold-col empty-col">-</td>
      `;
    }

    tr.innerHTML = `
      <td class="th-fixed"><strong>${item.test_case}</strong></td>
      <td class="th-fixed">${modelBadge}</td>
      <td class="th-fixed">${archBadge}</td>
      ${adaptedHtml}
      ${coldHtml}
    `;

    tbody.appendChild(tr);
  });
}

function exportResultsCsv() {
  const metrics = resultsData.metrics || [];
  if (metrics.length === 0) {
    showToast('No metrics to export', 'error');
    return;
  }

  // Group into side-by-side rows
  const grouped = {};
  metrics.forEach(m => {
    const key = `${m.test_case}__${m.model}__${m.architecture}`;
    if (!grouped[key]) {
      grouped[key] = { test_case: m.test_case, model: m.model, architecture: m.architecture, adapted: null, cold: null };
    }
    const mode = (m.mode || '').toLowerCase();
    if (mode.includes('fast') || mode.includes('adapt')) {
      grouped[key].adapted = m;
    } else {
      grouped[key].cold = m;
    }
  });

  const headers = [
    'Test Case', 'Model', 'Architecture',
    'Adapted Acc (%)', 'Adapted Macro F1 (%)', 'Adapted C0 Prec (%)', 'Adapted C0 Rec (%)', 'Adapted C1 Prec (%)', 'Adapted C1 Rec (%)', 'Adapted C1 F1 (%)', 'Adapted Confusion Matrix',
    'Cold-Start Acc (%)', 'Cold-Start Macro F1 (%)', 'Cold-Start C0 Prec (%)', 'Cold-Start C0 Rec (%)', 'Cold-Start C1 Prec (%)', 'Cold-Start C1 Rec (%)', 'Cold-Start C1 F1 (%)', 'Cold-Start Confusion Matrix'
  ];

  const rows = Object.values(grouped).map(item => {
    const ad = item.adapted;
    const cd = item.cold;

    const adC0 = ad ? (ad.class_0 || {}) : {};
    const adC1 = ad ? (ad.class_1 || {}) : {};
    const adCm = ad ? JSON.stringify(ad.confusion_matrix || []) : '""';

    const cdC0 = cd ? (cd.class_0 || {}) : {};
    const cdC1 = cd ? (cd.class_1 || {}) : {};
    const cdCm = cd ? JSON.stringify(cd.confusion_matrix || []) : '""';

    return [
      `"${item.test_case}"`,
      `"${item.model}"`,
      `"${item.architecture}"`,
      ad ? ad.accuracy : '',
      ad ? ad.macro_f1 : '',
      adC0.precision !== undefined ? adC0.precision : '',
      adC0.recall !== undefined ? adC0.recall : '',
      adC1.precision !== undefined ? adC1.precision : '',
      adC1.recall !== undefined ? adC1.recall : '',
      adC1.f1 !== undefined ? adC1.f1 : '',
      `"${adCm}"`,
      cd ? cd.accuracy : '',
      cd ? cd.macro_f1 : '',
      cdC0.precision !== undefined ? cdC0.precision : '',
      cdC0.recall !== undefined ? cdC0.recall : '',
      cdC1.precision !== undefined ? cdC1.precision : '',
      cdC1.recall !== undefined ? cdC1.recall : '',
      cdC1.f1 !== undefined ? cdC1.f1 : '',
      `"${cdCm}"`
    ].join(',');
  });

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `meta_learning_side_by_side_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('Side-by-side CSV exported successfully!', 'success');
}

function exportResultsJson() {
  const metrics = resultsData.metrics || [];
  if (metrics.length === 0) {
    showToast('No metrics to export', 'error');
    return;
  }

  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(resultsData, null, 2));
  const link = document.createElement("a");
  link.setAttribute("href", dataStr);
  link.setAttribute("download", `pipeline_results_${Date.now()}.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('JSON exported successfully!', 'success');
}

// ----------------------------------------------------------------------
// 5. Configuration (config.json) Management
// ----------------------------------------------------------------------
// Factory baseline default values (original system baseline with decoupled stages)
const FACTORY_DEFAULT_CONFIG = {
  constant: {
    steps_per_day: 144,
    note: "1 day = 144 time steps (24 hours at 10-minute intervals). Ground truth constant."
  },
  preprocessing: {
    raw_dir: "tabular/data/raw",
    output_dir: "tabular/data/processed",
    primary_csv: "dataset_prepro_routine_generated.csv",
    seed: 42,
    train_frac: 0.85,
    val_frac: 0.15,
    test_frac: 0,
    support_days: 1,
    tsls_cap: 1220.0
  },
  fcnet: {
    num_hidden_units: [40, 40],
    activation: "relu",
    dropout_rate: 0.0,
    use_layernorm: false
  },
  meta_training: {
    device: "cuda",
    num_epochs: 20,
    num_episodes_per_epoch: 1000,
    minibatch: 5,
    minibatch_print: 250,
    meta_lr: 0.001,
    inner_lr: 0.01,
    num_inner_updates: 5,
    first_order: true,
    num_models: 4,
    KL_weight: 0.000001,
    classification_threshold: 0.5,
    svgd_bandwidth_scale: 1.0,
    svgd_repulsive_weight: 1.0,
    gamma_prior_concentration: 1.0,
    gamma_prior_rate: 0.01,
    normal_prior_loc: 0.0,
    normal_prior_scale: 1.0,
    focal_loss: {
      alpha: 0.25,
      gamma: 2.0
    }
  },
  fast_adaptation_eval: {
    device: "cuda",
    checkpoint_mode: "best",
    support_days: 1,
    inner_lr: 0.01,
    num_inner_updates: 5,
    num_models: 4,
    KL_weight: 0.000001,
    classification_threshold: 0.5,
    first_order: true,
    svgd_bandwidth_scale: 1.0,
    svgd_repulsive_weight: 1.0,
    gamma_prior_concentration: 1.0,
    gamma_prior_rate: 0.01,
    normal_prior_loc: 0.0,
    normal_prior_scale: 1.0,
    focal_loss: {
      alpha: 0.25,
      gamma: 2.0
    }
  },
  cold_start_eval: {
    device: "cuda",
    checkpoint_mode: "best",
    support_days: 1,
    num_models: 4,
    classification_threshold: 0.5,
    gamma_prior_concentration: 1.0,
    gamma_prior_rate: 0.01,
    normal_prior_loc: 0.0,
    normal_prior_scale: 1.0,
    focal_loss: {
      alpha: 0.25,
      gamma: 2.0
    }
  }
};

async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    configData = await res.json();
    populateConfigForm(configData);
  } catch (err) {
    showToast(`Failed to load config.json: ${err.message}`, 'error');
  }
}

function populateConfigForm(cfg) {
  // 1. Preprocessing
  const p = cfg.preprocessing || cfg.dataset || {};
  const split = cfg.meta_training_split || {};
  document.getElementById('cfgPreproPrimaryCsv').value = p.primary_csv || 'dataset_prepro_routine_generated.csv';
  document.getElementById('cfgPreproSeed').value = p.seed !== undefined ? p.seed : 42;
  document.getElementById('cfgPreproSupportDays').value = p.support_days !== undefined ? p.support_days : (split.support_days !== undefined ? split.support_days : 1);
  document.getElementById('cfgPreproTrainFrac').value = p.train_frac !== undefined ? p.train_frac : (split.train_frac !== undefined ? split.train_frac : 0.85);
  document.getElementById('cfgPreproValFrac').value = p.val_frac !== undefined ? p.val_frac : (split.val_frac !== undefined ? split.val_frac : 0.15);
  document.getElementById('cfgPreproTestFrac').value = p.test_frac !== undefined ? p.test_frac : (split.test_frac !== undefined ? split.test_frac : 0);

  const elPreproTsls = document.getElementById('cfgPreproTslsCap');
  if (elPreproTsls) elPreproTsls.value = p.tsls_cap !== undefined ? p.tsls_cap : 1220.0;

  // 2. FcNet Architecture
  const fc = cfg.fcnet || {};
  const hiddenUnits = Array.isArray(fc.num_hidden_units) ? fc.num_hidden_units.join(', ') : (fc.num_hidden_units || '40, 40');
  const elHidden = document.getElementById('cfgFcnetHiddenUnits');
  if (elHidden) elHidden.value = hiddenUnits;

  const elAct = document.getElementById('cfgFcnetActivation');
  if (elAct) elAct.value = fc.activation || 'relu';

  const elDrop = document.getElementById('cfgFcnetDropout');
  if (elDrop) elDrop.value = fc.dropout_rate !== undefined ? fc.dropout_rate : 0.0;

  const chkLayernorm = document.getElementById('cfgFcnetLayernorm');
  if (chkLayernorm) {
    chkLayernorm.checked = Boolean(fc.use_layernorm);
    updateFcnetLayernormLabel();
  }

  // 3. Meta-Training
  const tr = cfg.meta_training || cfg.training_hyperparameters || {};
  const trFocal = tr.focal_loss || {};
  const elTrainDev = document.getElementById('cfgTrainDevice');
  if (elTrainDev) elTrainDev.value = tr.device || 'cuda';

  document.getElementById('cfgTrainNumEpochs').value = tr.num_epochs !== undefined ? tr.num_epochs : 20;
  document.getElementById('cfgTrainEpsPerEpoch').value = tr.num_episodes_per_epoch !== undefined ? tr.num_episodes_per_epoch : 1000;
  document.getElementById('cfgTrainMinibatch').value = tr.minibatch !== undefined ? tr.minibatch : 5;
  document.getElementById('cfgTrainMinibatchPrint').value = tr.minibatch_print !== undefined ? tr.minibatch_print : 250;
  document.getElementById('cfgTrainMetaLr').value = tr.meta_lr !== undefined ? tr.meta_lr : 0.001;
  document.getElementById('cfgTrainInnerUpdates').value = tr.num_inner_updates !== undefined ? tr.num_inner_updates : 5;
  document.getElementById('cfgTrainInnerLr').value = tr.inner_lr !== undefined ? tr.inner_lr : 0.01;
  document.getElementById('cfgTrainNumModels').value = tr.num_models !== undefined ? tr.num_models : 4;
  document.getElementById('cfgTrainKlWeight').value = tr.KL_weight !== undefined ? tr.KL_weight : 0.000001;
  const elTrainThresh = document.getElementById('cfgTrainThreshold');
  if (elTrainThresh) elTrainThresh.value = tr.classification_threshold !== undefined ? tr.classification_threshold : 0.5;
  document.getElementById('cfgTrainFocalAlpha').value = trFocal.alpha !== undefined ? trFocal.alpha : 0.25;
  document.getElementById('cfgTrainFocalGamma').value = trFocal.gamma !== undefined ? trFocal.gamma : 2.0;

  const elTrainSvgdBandwidth = document.getElementById('cfgTrainSvgdBandwidth');
  if (elTrainSvgdBandwidth) elTrainSvgdBandwidth.value = tr.svgd_bandwidth_scale !== undefined ? tr.svgd_bandwidth_scale : 1.0;
  const elTrainSvgdRepulsive = document.getElementById('cfgTrainSvgdRepulsive');
  if (elTrainSvgdRepulsive) elTrainSvgdRepulsive.value = tr.svgd_repulsive_weight !== undefined ? tr.svgd_repulsive_weight : 1.0;
  const elTrainGammaConc = document.getElementById('cfgTrainGammaConcentration');
  if (elTrainGammaConc) elTrainGammaConc.value = tr.gamma_prior_concentration !== undefined ? tr.gamma_prior_concentration : 1.0;
  const elTrainGammaRate = document.getElementById('cfgTrainGammaRate');
  if (elTrainGammaRate) elTrainGammaRate.value = tr.gamma_prior_rate !== undefined ? tr.gamma_prior_rate : 0.01;
  const elTrainNormLoc = document.getElementById('cfgTrainNormLoc');
  if (elTrainNormLoc) elTrainNormLoc.value = tr.normal_prior_loc !== undefined ? tr.normal_prior_loc : 0.0;
  const elTrainNormScale = document.getElementById('cfgTrainNormScale');
  if (elTrainNormScale) elTrainNormScale.value = tr.normal_prior_scale !== undefined ? tr.normal_prior_scale : 1.0;

  const chkTrainFirstOrder = document.getElementById('cfgTrainFirstOrder');
  if (chkTrainFirstOrder) {
    chkTrainFirstOrder.checked = (tr.first_order !== false);
    updateTrainFirstOrderLabel();
  }

  // 4. Fast Adaptation Eval
  const fa = cfg.fast_adaptation_eval || cfg.test_cases || {};
  const faFocal = fa.focal_loss || trFocal || {};
  const elAdaptDev = document.getElementById('cfgAdaptDevice');
  if (elAdaptDev) elAdaptDev.value = fa.device || 'cuda';

  const adaptCkptMode = fa.checkpoint_mode || 'best';
  const rdoAdaptBest = document.getElementById('cfgAdaptCkptBest');
  const rdoAdaptLast = document.getElementById('cfgAdaptCkptLast');
  const lblAdaptBest = document.getElementById('lblAdaptCkptBest');
  const lblAdaptLast = document.getElementById('lblAdaptCkptLast');
  if (adaptCkptMode === 'last') {
    if (rdoAdaptLast) rdoAdaptLast.checked = true;
    if (lblAdaptLast) lblAdaptLast.classList.add('active');
    if (lblAdaptBest) lblAdaptBest.classList.remove('active');
  } else {
    if (rdoAdaptBest) rdoAdaptBest.checked = true;
    if (lblAdaptBest) lblAdaptBest.classList.add('active');
    if (lblAdaptLast) lblAdaptLast.classList.remove('active');
  }

  document.getElementById('cfgAdaptSupportDays').value = fa.support_days !== undefined ? fa.support_days : 1;
  document.getElementById('cfgAdaptInnerUpdates').value = fa.num_inner_updates !== undefined ? fa.num_inner_updates : (tr.num_inner_updates || 5);
  document.getElementById('cfgAdaptInnerLr').value = fa.inner_lr !== undefined ? fa.inner_lr : (tr.inner_lr || 0.01);
  document.getElementById('cfgAdaptNumModels').value = fa.num_models !== undefined ? fa.num_models : (tr.num_models || 4);
  const elAdaptKl = document.getElementById('cfgAdaptKlWeight');
  if (elAdaptKl) elAdaptKl.value = fa.KL_weight !== undefined ? fa.KL_weight : (tr.KL_weight !== undefined ? tr.KL_weight : 0.000001);
  const elAdaptThresh = document.getElementById('cfgAdaptThreshold');
  if (elAdaptThresh) elAdaptThresh.value = fa.classification_threshold !== undefined ? fa.classification_threshold : 0.5;
  document.getElementById('cfgAdaptFocalAlpha').value = faFocal.alpha !== undefined ? faFocal.alpha : 0.25;
  document.getElementById('cfgAdaptFocalGamma').value = faFocal.gamma !== undefined ? faFocal.gamma : 2.0;

  const elAdaptSvgdBandwidth = document.getElementById('cfgAdaptSvgdBandwidth');
  if (elAdaptSvgdBandwidth) elAdaptSvgdBandwidth.value = fa.svgd_bandwidth_scale !== undefined ? fa.svgd_bandwidth_scale : 1.0;
  const elAdaptSvgdRepulsive = document.getElementById('cfgAdaptSvgdRepulsive');
  if (elAdaptSvgdRepulsive) elAdaptSvgdRepulsive.value = fa.svgd_repulsive_weight !== undefined ? fa.svgd_repulsive_weight : 1.0;

  const chkAdaptFirstOrder = document.getElementById('cfgAdaptFirstOrder');
  if (chkAdaptFirstOrder) {
    chkAdaptFirstOrder.checked = (fa.first_order !== false);
    updateAdaptFirstOrderLabel();
  }

  // 5. Cold-Start Eval
  const cs = cfg.cold_start_eval || {};
  const csFocal = cs.focal_loss || trFocal || {};
  const elColdDev = document.getElementById('cfgColdDevice');
  if (elColdDev) elColdDev.value = cs.device || 'cuda';

  const coldCkptMode = cs.checkpoint_mode || 'best';
  const rdoColdBest = document.getElementById('cfgColdCkptBest');
  const rdoColdLast = document.getElementById('cfgColdCkptLast');
  const lblColdBest = document.getElementById('lblColdCkptBest');
  const lblColdLast = document.getElementById('lblColdCkptLast');
  if (coldCkptMode === 'last') {
    if (rdoColdLast) rdoColdLast.checked = true;
    if (lblColdLast) lblColdLast.classList.add('active');
    if (lblColdBest) lblColdBest.classList.remove('active');
  } else {
    if (rdoColdBest) rdoColdBest.checked = true;
    if (lblColdBest) lblColdBest.classList.add('active');
    if (lblColdLast) lblColdLast.classList.remove('active');
  }

  document.getElementById('cfgColdSupportDays').value = cs.support_days !== undefined ? cs.support_days : 1;
  document.getElementById('cfgColdNumModels').value = cs.num_models !== undefined ? cs.num_models : (tr.num_models || 4);
  const elColdThresh = document.getElementById('cfgColdThreshold');
  if (elColdThresh) elColdThresh.value = cs.classification_threshold !== undefined ? cs.classification_threshold : 0.5;
  document.getElementById('cfgColdFocalAlpha').value = csFocal.alpha !== undefined ? csFocal.alpha : 0.25;
  document.getElementById('cfgColdFocalGamma').value = csFocal.gamma !== undefined ? csFocal.gamma : 2.0;

  updateSplitVisualizer();
}

function updateFcnetLayernormLabel() {
  const chk = document.getElementById('cfgFcnetLayernorm');
  const lbl = document.getElementById('lblFcnetLayernorm');
  if (chk && lbl) {
    lbl.textContent = chk.checked ? 'Enabled' : 'Disabled';
  }
}

function updateTrainFirstOrderLabel() {
  const chk = document.getElementById('cfgTrainFirstOrder');
  const lbl = document.getElementById('lblTrainFirstOrder');
  if (chk && lbl) {
    lbl.textContent = chk.checked ? 'Enabled' : 'Full Hessian (Slower)';
  }
}

function updateAdaptFirstOrderLabel() {
  const chk = document.getElementById('cfgAdaptFirstOrder');
  const lbl = document.getElementById('lblAdaptFirstOrder');
  if (chk && lbl) {
    lbl.textContent = chk.checked ? 'Enabled' : 'Full Hessian';
  }
}

function getNumberVal(id, defaultVal) {
  const el = document.getElementById(id);
  if (!el || el.value === '' || el.value === null) return defaultVal;
  const num = Number(el.value);
  return isNaN(num) ? defaultVal : num;
}

function updateSplitVisualizer() {
  const train = getNumberVal('cfgPreproTrainFrac', 0.85);
  const val = getNumberVal('cfgPreproValFrac', 0.15);
  const test = getNumberVal('cfgPreproTestFrac', 0.0);

  const total = train + val + test;
  const barTrain = document.getElementById('splitBarTrain');
  const barVal = document.getElementById('splitBarVal');
  const barTest = document.getElementById('splitBarTest');

  if (!barTrain || !barVal || !barTest) return;

  if (total <= 0) {
    barTrain.style.width = '0%';
    barTrain.textContent = '0%';
    barVal.style.width = '0%';
    barVal.textContent = '0%';
    barTest.style.width = '0%';
    barTest.textContent = '0%';
    return;
  }

  const trainPct = Math.round((train / total) * 100);
  const valPct = Math.round((val / total) * 100);
  const testPct = Math.max(0, 100 - trainPct - valPct);

  barTrain.style.width = `${trainPct}%`;
  barTrain.textContent = `Train ${trainPct}%`;

  barVal.style.width = `${valPct}%`;
  barVal.textContent = `Val ${valPct}%`;

  barTest.style.width = `${testPct}%`;
  barTest.textContent = `Test ${testPct}%`;
}

['cfgPreproTrainFrac', 'cfgPreproValFrac', 'cfgPreproTestFrac'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', updateSplitVisualizer);
});

const chkFcnetLayernormEl = document.getElementById('cfgFcnetLayernorm');
if (chkFcnetLayernormEl) {
  chkFcnetLayernormEl.addEventListener('change', updateFcnetLayernormLabel);
}

const chkTrainFirstOrderEl = document.getElementById('cfgTrainFirstOrder');
if (chkTrainFirstOrderEl) {
  chkTrainFirstOrderEl.addEventListener('change', updateTrainFirstOrderLabel);
}

function setupCheckpointRadios(bestId, lastId, lblBestId, lblLastId) {
  const rdoBest = document.getElementById(bestId);
  const rdoLast = document.getElementById(lastId);
  const lblBest = document.getElementById(lblBestId);
  const lblLast = document.getElementById(lblLastId);

  if (lblBest && rdoBest) {
    lblBest.addEventListener('click', (e) => {
      e.preventDefault();
      rdoBest.checked = true;
      lblBest.classList.add('active');
      if (lblLast) lblLast.classList.remove('active');
    });
  }
  if (lblLast && rdoLast) {
    lblLast.addEventListener('click', (e) => {
      e.preventDefault();
      rdoLast.checked = true;
      lblLast.classList.add('active');
      if (lblBest) lblBest.classList.remove('active');
    });
  }
}
setupCheckpointRadios('cfgAdaptCkptBest', 'cfgAdaptCkptLast', 'lblAdaptCkptBest', 'lblAdaptCkptLast');
setupCheckpointRadios('cfgColdCkptBest', 'cfgColdCkptLast', 'lblColdCkptBest', 'lblColdCkptLast');

async function saveConfig() {
  const hiddenUnitsStr = document.getElementById('cfgFcnetHiddenUnits') ? document.getElementById('cfgFcnetHiddenUnits').value : '40, 40';
  const parsedHiddenUnits = hiddenUnitsStr
    .split(',')
    .map(s => parseInt(s.trim(), 10))
    .filter(n => !isNaN(n) && n > 0);

  configData = {
    constant: {
      steps_per_day: 144,
      note: "1 day = 144 time steps (24 hours at 10-minute intervals). Ground truth constant."
    },
    preprocessing: {
      raw_dir: "tabular/data/raw",
      output_dir: "tabular/data/processed",
      primary_csv: document.getElementById('cfgPreproPrimaryCsv').value || 'dataset_prepro_routine_generated.csv',
      seed: getNumberVal('cfgPreproSeed', 42),
      train_frac: getNumberVal('cfgPreproTrainFrac', 0.85),
      val_frac: getNumberVal('cfgPreproValFrac', 0.15),
      test_frac: getNumberVal('cfgPreproTestFrac', 0),
      support_days: getNumberVal('cfgPreproSupportDays', 1),
      tsls_cap: getNumberVal('cfgPreproTslsCap', 1220.0)
    },
    fcnet: {
      num_hidden_units: parsedHiddenUnits.length > 0 ? parsedHiddenUnits : [40, 40],
      activation: document.getElementById('cfgFcnetActivation') ? document.getElementById('cfgFcnetActivation').value : 'relu',
      dropout_rate: getNumberVal('cfgFcnetDropout', 0.0),
      use_layernorm: document.getElementById('cfgFcnetLayernorm') ? document.getElementById('cfgFcnetLayernorm').checked : false
    },
    meta_training: {
      device: document.getElementById('cfgTrainDevice') ? document.getElementById('cfgTrainDevice').value : "cuda",
      num_epochs: getNumberVal('cfgTrainNumEpochs', 20),
      num_episodes_per_epoch: getNumberVal('cfgTrainEpsPerEpoch', 1000),
      minibatch: getNumberVal('cfgTrainMinibatch', 5),
      minibatch_print: getNumberVal('cfgTrainMinibatchPrint', 250),
      meta_lr: getNumberVal('cfgTrainMetaLr', 0.001),
      inner_lr: getNumberVal('cfgTrainInnerLr', 0.01),
      num_inner_updates: getNumberVal('cfgTrainInnerUpdates', 5),
      first_order: document.getElementById('cfgTrainFirstOrder') ? document.getElementById('cfgTrainFirstOrder').checked : true,
      num_models: getNumberVal('cfgTrainNumModels', 4),
      KL_weight: getNumberVal('cfgTrainKlWeight', 0.000001),
      classification_threshold: getNumberVal('cfgTrainThreshold', 0.5),
      svgd_bandwidth_scale: getNumberVal('cfgTrainSvgdBandwidth', 1.0),
      svgd_repulsive_weight: getNumberVal('cfgTrainSvgdRepulsive', 1.0),
      gamma_prior_concentration: getNumberVal('cfgTrainGammaConcentration', 1.0),
      gamma_prior_rate: getNumberVal('cfgTrainGammaRate', 0.01),
      normal_prior_loc: getNumberVal('cfgTrainNormLoc', 0.0),
      normal_prior_scale: getNumberVal('cfgTrainNormScale', 1.0),
      focal_loss: {
        alpha: getNumberVal('cfgTrainFocalAlpha', 0.25),
        gamma: getNumberVal('cfgTrainFocalGamma', 2.0)
      }
    },
    fast_adaptation_eval: {
      device: document.getElementById('cfgAdaptDevice') ? document.getElementById('cfgAdaptDevice').value : "cuda",
      checkpoint_mode: document.getElementById('cfgAdaptCkptLast') && document.getElementById('cfgAdaptCkptLast').checked ? 'last' : 'best',
      support_days: getNumberVal('cfgAdaptSupportDays', 1),
      inner_lr: getNumberVal('cfgAdaptInnerLr', 0.01),
      num_inner_updates: getNumberVal('cfgAdaptInnerUpdates', 5),
      num_models: getNumberVal('cfgAdaptNumModels', 4),
      KL_weight: getNumberVal('cfgAdaptKlWeight', 0.000001),
      classification_threshold: getNumberVal('cfgAdaptThreshold', 0.5),
      first_order: document.getElementById('cfgAdaptFirstOrder') ? document.getElementById('cfgAdaptFirstOrder').checked : true,
      svgd_bandwidth_scale: getNumberVal('cfgAdaptSvgdBandwidth', 1.0),
      svgd_repulsive_weight: getNumberVal('cfgAdaptSvgdRepulsive', 1.0),
      gamma_prior_concentration: getNumberVal('cfgTrainGammaConcentration', 1.0),
      gamma_prior_rate: getNumberVal('cfgTrainGammaRate', 0.01),
      normal_prior_loc: getNumberVal('cfgTrainNormLoc', 0.0),
      normal_prior_scale: getNumberVal('cfgTrainNormScale', 1.0),
      focal_loss: {
        alpha: getNumberVal('cfgAdaptFocalAlpha', 0.25),
        gamma: getNumberVal('cfgAdaptFocalGamma', 2.0)
      }
    },
    cold_start_eval: {
      device: document.getElementById('cfgColdDevice') ? document.getElementById('cfgColdDevice').value : "cuda",
      checkpoint_mode: document.getElementById('cfgColdCkptLast') && document.getElementById('cfgColdCkptLast').checked ? 'last' : 'best',
      support_days: getNumberVal('cfgColdSupportDays', 1),
      num_models: getNumberVal('cfgColdNumModels', 4),
      classification_threshold: getNumberVal('cfgColdThreshold', 0.5),
      gamma_prior_concentration: getNumberVal('cfgTrainGammaConcentration', 1.0),
      gamma_prior_rate: getNumberVal('cfgTrainGammaRate', 0.01),
      normal_prior_loc: getNumberVal('cfgTrainNormLoc', 0.0),
      normal_prior_scale: getNumberVal('cfgTrainNormScale', 1.0),
      focal_loss: {
        alpha: getNumberVal('cfgColdFocalAlpha', 0.25),
        gamma: getNumberVal('cfgColdFocalGamma', 2.0)
      }
    }
  };

  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(configData)
    });
    const data = await res.json();
    showToast('Saved and applied decoupled configuration to config.json!', 'success');
  } catch (err) {
    showToast(`Failed to save config: ${err.message}`, 'error');
  }
}

async function resetToFactoryDefaults() {
  if (!confirm('Are you sure you want to reset all hyperparameters to original factory defaults?')) {
    return;
  }

  configData = JSON.parse(JSON.stringify(FACTORY_DEFAULT_CONFIG));
  populateConfigForm(configData);

  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(configData)
    });
    const data = await res.json();
    showToast('Reset all parameters to factory defaults in config.json!', 'success');
  } catch (err) {
    showToast(`Failed to reset config: ${err.message}`, 'error');
  }
}

// ----------------------------------------------------------------------
// 6. Pipeline Execution & Status Polling
// ----------------------------------------------------------------------
function initActions() {
  document.getElementById('btnSaveConfig').addEventListener('click', saveConfig);
  document.getElementById('btnResetConfig').addEventListener('click', resetToFactoryDefaults);

  document.getElementById('btnRunPipeline').addEventListener('click', () => runPipeline(false));
  document.getElementById('btnDryRun').addEventListener('click', () => runPipeline(true));
  document.getElementById('btnStopPipeline').addEventListener('click', stopPipeline);

  document.getElementById('btnClearTerminal').addEventListener('click', async () => {
    document.getElementById('terminalOutput').textContent = 'Console logs cleared.';
    try {
      await fetch('/api/terminal/clear', { method: 'POST' });
      showToast('Console logs cleared successfully!', 'success');
    } catch (e) {
      showToast('Cleared local terminal console', 'success');
    }
  });

  document.getElementById('btnCopyTerminal').addEventListener('click', () => {
    const text = document.getElementById('terminalOutput').textContent;
    navigator.clipboard.writeText(text);
    showToast('Logs copied to clipboard!', 'success');
  });
}

async function runPipeline(isDryRun = false) {
  try {
    // Switch to Results tab automatically
    document.querySelector('.nav-item[data-tab="results-tab"]').click();

    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dry_run: isDryRun })
    });
    const data = await res.json();

    if (data.status === 'started') {
      showToast(isDryRun ? 'Starting dry-run preview...' : 'Pipeline execution started!', 'success');
    } else {
      showToast(data.message || 'Execution error', 'error');
    }
  } catch (err) {
    showToast(`Run error: ${err.message}`, 'error');
  }
}

async function stopPipeline() {
  try {
    const res = await fetch('/api/stop', { method: 'POST' });
    const data = await res.json();
    showToast(data.message || 'Pipeline stopped', 'error');
  } catch (err) {
    showToast(`Stop error: ${err.message}`, 'error');
  }
}

function startStatusPolling() {
  if (pollingInterval) clearInterval(pollingInterval);

  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const status = await res.json();

      updateExecutionUI(status);
    } catch (e) {
      // Ignore network blips
    }
  }, 1000);
}

function updateExecutionUI(status) {
  const ind = document.getElementById('globalStatusIndicator');
  const title = document.getElementById('globalStatusText');
  const detail = document.getElementById('globalStatusDetail');
  const terminal = document.getElementById('terminalOutput');
  const autoScroll = document.getElementById('chkAutoScroll').checked;
  const liveDot = document.getElementById('resultsLiveBadge');

  const btnRun = document.getElementById('btnRunPipeline');
  const btnDry = document.getElementById('btnDryRun');
  const btnStop = document.getElementById('btnStopPipeline');

  isCurrentlyRunning = status.running;

  if (status.running) {
    ind.className = 'status-indicator running';
    title.textContent = `Running (${status.elapsed_seconds}s)`;
    detail.textContent = `Active pipeline jobs running`;
    liveDot.classList.add('active');

    btnRun.classList.add('hidden');
    btnDry.classList.add('hidden');
    btnStop.classList.remove('hidden');
  } else {
    ind.className = status.exit_code === 0 ? 'status-indicator idle' : (status.exit_code ? 'status-indicator error' : 'status-indicator idle');
    title.textContent = status.exit_code === 0 ? 'Ready (Completed)' : (status.exit_code ? 'Failed (Exit code != 0)' : 'Ready (Idle)');
    detail.textContent = status.exit_code !== null ? `Last run finished with code ${status.exit_code}` : 'No active job running';
    liveDot.classList.remove('active');

    btnRun.classList.remove('hidden');
    btnDry.classList.remove('hidden');
    btnStop.classList.add('hidden');
  }

  // Update tasks cards and table
  if (status.tasks_status) {
    resultsData.tasks_status = status.tasks_status;
    renderTasksStatus(status.tasks_status);
  }
  if (status.metrics && status.metrics.length > 0) {
    resultsData.metrics = status.metrics;
    renderComparisonTable(status.metrics);
  }

  // Update raw console terminal
  if (status.logs && status.logs.length > 0) {
    terminal.textContent = status.logs;
    if (autoScroll) {
      terminal.scrollTop = terminal.scrollHeight;
    }
  }
}

// ----------------------------------------------------------------------
// 7. Toast System
// ----------------------------------------------------------------------
function showToast(message, type = 'success') {
  const toast = document.getElementById('toastNotification');
  toast.textContent = message;
  toast.className = `toast show ${type}`;

  setTimeout(() => {
    toast.className = 'toast';
  }, 3500);
}
