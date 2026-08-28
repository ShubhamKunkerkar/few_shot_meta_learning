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
let activeModeFilter = 'all';

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
    const data = await res.json();
    if (showNotification) {
      showToast('Pipeline order and toggles saved successfully!', 'success');
    }
  } catch (err) {
    showToast(`Failed to save pipeline: ${err.message}`, 'error');
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
  document.querySelectorAll('#modeFilterPills .pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#modeFilterPills .pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      activeModeFilter = btn.getAttribute('data-filter-mode');
      renderComparisonTable(resultsData.metrics || []);
    });
  });

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

function sortedUnique(arr) {
  return Array.from(new Set(arr.filter(Boolean))).sort();
}

function renderComparisonTable(metrics) {
  updateTestCaseFilterPills(metrics);

  const tbody = document.getElementById('metricsTableBody');
  tbody.innerHTML = '';

  if (!metrics || metrics.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-table-msg">No evaluation metrics recorded yet.</td></tr>';
    return;
  }

  // Filter metrics
  let filtered = metrics;
  if (activeTcFilter !== 'all') {
    filtered = filtered.filter(m => m.test_case === activeTcFilter);
  }
  if (activeModeFilter !== 'all') {
    filtered = filtered.filter(m => m.mode === activeModeFilter);
  }

  if (filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-table-msg">No metrics match the current filter selection.</td></tr>';
    return;
  }

  // Find max accuracy and F1 for highlighting
  const maxAcc = Math.max(...filtered.map(m => m.accuracy || 0));
  const maxF1 = Math.max(...filtered.map(m => m.macro_f1 || 0));

  filtered.forEach(m => {
    const tr = document.createElement('tr');

    const isTopAcc = (m.accuracy === maxAcc && maxAcc > 0);
    const isTopF1 = (m.macro_f1 === maxF1 && maxF1 > 0);

    const modelBadge = `<span class="badge badge-${(m.model || '').toLowerCase()}">${m.model}</span>`;
    const modeBadge = `<span class="mode-pill ${m.mode === 'Fast Adaptation' ? 'mode-adapted' : 'mode-cold'}">${m.mode}</span>`;

    const c1 = m.class_1 || {};
    const pr1Str = `${c1.precision !== undefined ? c1.precision.toFixed(1) : '0.0'}% / ${c1.recall !== undefined ? c1.recall.toFixed(1) : '0.0'}%`;

    const cm = m.confusion_matrix || [[0, 0], [0, 0]];
    const cmStr = `[${cm[0][0]}, ${cm[0][1]} / ${cm[1][0]}, ${cm[1][1]}]`;

    tr.innerHTML = `
      <td><strong>${m.test_case}</strong></td>
      <td>${modelBadge}</td>
      <td><span class="badge badge-arch">${m.architecture}</span></td>
      <td>${modeBadge}</td>
      <td><span class="metric-num ${isTopAcc ? 'metric-highlight' : ''}">${m.accuracy.toFixed(2)}% ${isTopAcc ? '⭐' : ''}</span></td>
      <td><span class="metric-num ${isTopF1 ? 'metric-highlight' : ''}">${m.macro_f1.toFixed(2)}%</span></td>
      <td><span class="metric-num">${c1.f1 !== undefined ? c1.f1.toFixed(2) : '0.00'}%</span></td>
      <td><span class="metric-num">${pr1Str}</span></td>
      <td><span class="cm-box">${cmStr}</span></td>
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

  const headers = ['Test Case', 'Model', 'Architecture', 'Mode', 'Accuracy (%)', 'Macro F1 (%)', 'Class 1 Precision (%)', 'Class 1 Recall (%)', 'Class 1 F1 (%)', 'Confusion Matrix'];
  const rows = metrics.map(m => {
    const c1 = m.class_1 || {};
    const cm = JSON.stringify(m.confusion_matrix || []);
    return [
      `"${m.test_case}"`,
      `"${m.model}"`,
      `"${m.architecture}"`,
      `"${m.mode}"`,
      m.accuracy,
      m.macro_f1,
      c1.precision || 0,
      c1.recall || 0,
      c1.f1 || 0,
      `"${cm}"`
    ].join(',');
  });

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `meta_learning_comparison_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('CSV exported successfully!', 'success');
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
// Factory baseline default values (original system baseline)
const FACTORY_DEFAULT_CONFIG = {
  dataset: {
    raw_dir: "tabular/data/raw",
    output_dir: "tabular/data/processed",
    primary_csv: "dataset_prepro_routine_generated.csv",
    steps_per_day: 144,
    seed: 42
  },
  meta_training_split: {
    train_frac: 0.70,
    val_frac: 0.15,
    test_frac: 0.15,
    support_days: 1,
    k_shot: 144
  },
  test_cases: {
    support_days: 1,
    k_shot: 144
  },
  training_hyperparameters: {
    num_epochs: 20,
    num_episodes_per_epoch: 1000,
    minibatch: 5,
    minibatch_print: 250,
    meta_lr: 0.001,
    first_order: true,
    num_inner_updates: 5,
    inner_lr: 0.01,
    KL_weight: 0.000001,
    num_models: 4,
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
  const d = cfg.dataset || {};
  const s = cfg.meta_training_split || {};
  const tc = cfg.test_cases || {};
  const h = cfg.training_hyperparameters || {};
  const f = h.focal_loss || {};

  document.getElementById('cfgPrimaryCsv').value = d.primary_csv || 'dataset_prepro_routine_generated.csv';
  document.getElementById('cfgStepsPerDay').value = d.steps_per_day !== undefined ? d.steps_per_day : 144;
  document.getElementById('cfgSeed').value = d.seed !== undefined ? d.seed : 42;

  document.getElementById('cfgTrainFrac').value = s.train_frac !== undefined ? s.train_frac : 0.70;
  document.getElementById('cfgValFrac').value = s.val_frac !== undefined ? s.val_frac : 0.15;
  document.getElementById('cfgTestFrac').value = s.test_frac !== undefined ? s.test_frac : 0.15;
  document.getElementById('cfgSupportDays').value = s.support_days !== undefined ? s.support_days : 1;
  document.getElementById('cfgKShot').value = s.k_shot !== undefined ? s.k_shot : 144;

  document.getElementById('cfgTcSupportDays').value = tc.support_days !== undefined ? tc.support_days : 1;
  document.getElementById('cfgTcKShot').value = tc.k_shot !== undefined ? tc.k_shot : 144;

  document.getElementById('cfgNumEpochs').value = h.num_epochs !== undefined ? h.num_epochs : 20;
  document.getElementById('cfgEpsPerEpoch').value = h.num_episodes_per_epoch !== undefined ? h.num_episodes_per_epoch : 1000;
  document.getElementById('cfgMinibatch').value = h.minibatch !== undefined ? h.minibatch : 5;
  document.getElementById('cfgMinibatchPrint').value = h.minibatch_print !== undefined ? h.minibatch_print : 250;
  document.getElementById('cfgMetaLr').value = h.meta_lr !== undefined ? h.meta_lr : 0.001;

  const chkFirstOrder = document.getElementById('cfgFirstOrder');
  if (chkFirstOrder) {
    chkFirstOrder.checked = (h.first_order !== false);
    updateFirstOrderLabel();
  }

  document.getElementById('cfgInnerUpdates').value = h.num_inner_updates !== undefined ? h.num_inner_updates : 5;
  document.getElementById('cfgInnerLr').value = h.inner_lr !== undefined ? h.inner_lr : 0.01;

  document.getElementById('cfgNumModels').value = h.num_models !== undefined ? h.num_models : 4;
  document.getElementById('cfgKlWeight').value = h.KL_weight !== undefined ? h.KL_weight : 0.000001;

  document.getElementById('cfgFocalAlpha').value = f.alpha !== undefined ? f.alpha : 0.25;
  document.getElementById('cfgFocalGamma').value = f.gamma !== undefined ? f.gamma : 2.0;

  updateSplitVisualizer();
}

function updateFirstOrderLabel() {
  const chk = document.getElementById('cfgFirstOrder');
  const lbl = document.getElementById('lblFirstOrder');
  if (chk && lbl) {
    lbl.textContent = chk.checked ? 'First-Order Approximation Enabled' : 'Full Second-Order Derivatives (Slower)';
  }
}

function updateSplitVisualizer() {
  const train = parseFloat(document.getElementById('cfgTrainFrac').value) || 0.70;
  const val = parseFloat(document.getElementById('cfgValFrac').value) || 0.15;
  const test = parseFloat(document.getElementById('cfgTestFrac').value) || 0.15;

  const total = train + val + test;
  const pTrain = Math.round((train / total) * 100);
  const pVal = Math.round((val / total) * 100);
  const pTest = 100 - pTrain - pVal;

  const barTrain = document.getElementById('splitBarTrain');
  const barVal = document.getElementById('splitBarVal');
  const barTest = document.getElementById('splitBarTest');

  if (barTrain && barVal && barTest) {
    barTrain.style.width = `${pTrain}%`;
    barTrain.textContent = `Train ${pTrain}%`;

    barVal.style.width = `${pVal}%`;
    barVal.textContent = `Val ${pVal}%`;

    barTest.style.width = `${pTest}%`;
    barTest.textContent = `Test ${pTest}%`;
  }
}

['cfgTrainFrac', 'cfgValFrac', 'cfgTestFrac'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', updateSplitVisualizer);
});

const chkFirstOrderEl = document.getElementById('cfgFirstOrder');
if (chkFirstOrderEl) {
  chkFirstOrderEl.addEventListener('change', updateFirstOrderLabel);
}

async function saveConfig() {
  configData = {
    dataset: {
      raw_dir: "tabular/data/raw",
      output_dir: "tabular/data/processed",
      primary_csv: document.getElementById('cfgPrimaryCsv').value || 'dataset_prepro_routine_generated.csv',
      steps_per_day: parseInt(document.getElementById('cfgStepsPerDay').value) || 144,
      seed: parseInt(document.getElementById('cfgSeed').value) || 42
    },
    meta_training_split: {
      train_frac: parseFloat(document.getElementById('cfgTrainFrac').value) || 0.70,
      val_frac: parseFloat(document.getElementById('cfgValFrac').value) || 0.15,
      test_frac: parseFloat(document.getElementById('cfgTestFrac').value) || 0.15,
      support_days: parseInt(document.getElementById('cfgSupportDays').value) || 1,
      k_shot: parseInt(document.getElementById('cfgKShot').value) || 144
    },
    test_cases: {
      support_days: parseInt(document.getElementById('cfgTcSupportDays').value) || 1,
      k_shot: parseInt(document.getElementById('cfgTcKShot').value) || 144
    },
    training_hyperparameters: {
      num_epochs: parseInt(document.getElementById('cfgNumEpochs').value) || 20,
      num_episodes_per_epoch: parseInt(document.getElementById('cfgEpsPerEpoch').value) || 1000,
      minibatch: parseInt(document.getElementById('cfgMinibatch').value) || 5,
      minibatch_print: parseInt(document.getElementById('cfgMinibatchPrint').value) || 250,
      meta_lr: parseFloat(document.getElementById('cfgMetaLr').value) || 0.001,
      first_order: document.getElementById('cfgFirstOrder') ? document.getElementById('cfgFirstOrder').checked : true,
      num_inner_updates: parseInt(document.getElementById('cfgInnerUpdates').value) || 5,
      inner_lr: parseFloat(document.getElementById('cfgInnerLr').value) || 0.01,
      KL_weight: parseFloat(document.getElementById('cfgKlWeight').value) || 0.000001,
      num_models: parseInt(document.getElementById('cfgNumModels').value) || 4,
      focal_loss: {
        alpha: parseFloat(document.getElementById('cfgFocalAlpha').value) || 0.25,
        gamma: parseFloat(document.getElementById('cfgFocalGamma').value) || 2.0
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
    showToast('Saved and applied configuration to config.json!', 'success');
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
    showToast('Reset all hyperparameters to original baseline defaults!', 'success');
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

  document.getElementById('btnClearTerminal').addEventListener('click', () => {
    document.getElementById('terminalOutput').textContent = '';
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
