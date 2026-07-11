const playerSearch = document.getElementById('player-search');
const positionCheckboxes = document.querySelectorAll('input[name="position"]');
const playerTbody = document.querySelector('#player-list tbody');
const newDraftBtn = document.getElementById('new-draft-btn-header');
const aiChip = document.getElementById('ai-chip');
const boardHeaders = document.getElementById('board-headers');
const boardRows = document.getElementById('board-rows');
const currentPickDisplay = document.getElementById('current-pick-display');
const teamOnClockDisplay = document.getElementById('team-on-clock-display');
const aiSuggestionDisplay = document.getElementById('ai-suggestion-display');
// We fetch AI suggestion only for the team currently picking
const totalPicksDisplay = document.getElementById('total-picks-display');
const qbDraftedDisplay = document.getElementById('qb-drafted-display');
const rbDraftedDisplay = document.getElementById('rb-drafted-display');
const wrDraftedDisplay = document.getElementById('wr-drafted-display');
const teDraftedDisplay = document.getElementById('te-drafted-display');
const undoPickBtn = document.getElementById('undo-pick-btn');
const simulatePickBtn = document.getElementById('simulate-pick-btn');
const autoDraftBtn = document.getElementById('auto-draft-btn');
const simModeSelect = document.getElementById('sim-mode-select');
const downloadCsvBtn = document.getElementById('download-csv-btn');
const AUTO_DRAFT_LABEL = 'Auto draft';
const gpFracMinInput = document.getElementById('gp-frac-min');
const simulateSeasonBtn = document.getElementById('simulate-season-btn');
const seasonResultsDiv = document.getElementById('season-simulation-results');
const insightModal = document.getElementById('insight-modal');
const insightModalTitle = document.getElementById('insight-modal-title');
const insightModalPhrase = document.getElementById('insight-modal-phrase');
const insightModalSummary = document.getElementById('insight-modal-summary');
const insightModalTags = document.getElementById('insight-modal-tags');
const insightModalBullets = document.getElementById('insight-modal-bullets');
const insightModalConfidence = document.getElementById('insight-modal-confidence');
const insightModalClose = document.getElementById('insight-modal-close');
const insightModalGoogle = document.getElementById('insight-modal-google');
const insightsHint = document.getElementById('insights-hint');
const outlookColHeader = document.getElementById('outlook-col-header');
const outlookTooltip = document.getElementById('outlook-tooltip');
const outlookTooltipPhrase = document.getElementById('outlook-tooltip-phrase');
const outlookTooltipRole = document.getElementById('outlook-tooltip-role');
const outlookTooltipSummary = document.getElementById('outlook-tooltip-summary');
const assistantAutoToggle = document.getElementById('assistant-auto-toggle');
const assistantScopeSelect = document.getElementById('assistant-scope-select');
const assistantAgentModelSelect = document.getElementById('assistant-agent-model-select');
const assistantOtherModelSelect = document.getElementById('assistant-other-model-select');
const askAssistantBtn = document.getElementById('ask-assistant-btn');
const assistantBanner = document.getElementById('assistant-banner');
const advisorPanel = document.getElementById('advisor-panel');
const advisorPanelTitle = document.getElementById('advisor-panel-title');
const advisorPanelBody = document.getElementById('advisor-panel-body');
const advisorPanelClose = document.getElementById('advisor-panel-close');

let modalPlayerName = null;

const topPanel = document.getElementById('top-panel');
const bottomPanel = document.getElementById('bottom-panel');
const resizer = document.getElementById('h-resizer');

let isResizing = false;

resizer.addEventListener('mousedown', function(e) {
    isResizing = true;
    document.body.style.cursor = 'row-resize';
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
});

function handleMouseMove(e) {
    if (!isResizing) return;
    const main = document.querySelector('main');
    const mainRect = main.getBoundingClientRect();
    const topHeight = e.clientY - mainRect.top;
    const containerHeight = mainRect.height;
    const percentage = (topHeight / containerHeight) * 100;
    if (percentage > 5 && percentage < 95) {
        topPanel.style.flex = `0 0 ${percentage}%`;
    }
}

function handleMouseUp(e) {
    isResizing = false;
    document.body.style.cursor = 'default';
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
}

let currentDraftState = null;

let currentSortBy = 'vorp';
let currentSortDir = 'desc'; // 'asc' or 'desc'
let lastRenderedPlayers = [];
// Track players to ignore for AI suggestion
const blindSet = new Set();

let advisorInFlight = false;
let lastAutoAdvisorPickNumber = null;
let previousPickNumber = null;
let advisorModelsLoaded = false;

function populateAdvisorModelSelect(selectElement, models, selectedValue) {
    if (!selectElement) return;
    selectElement.innerHTML = '';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.label;
        selectElement.appendChild(option);
    });
    if (selectedValue && models.some(model => model.id === selectedValue)) {
        selectElement.value = selectedValue;
    } else if (models.length > 0) {
        selectElement.value = models[0].id;
    }
}

async function loadAdvisorModels() {
    if (!assistantAgentModelSelect || !assistantOtherModelSelect) return;
    try {
        const response = await fetch('/api/draft/advisor/models');
        if (!response.ok) {
            assistantAgentModelSelect.disabled = true;
            assistantOtherModelSelect.disabled = true;
            if (askAssistantBtn) askAssistantBtn.disabled = true;
            return;
        }
        const data = await response.json();
        const models = data.models || [];
        const storedAgentModel = sessionStorage.getItem('assistantAgentModel');
        const storedOtherModel = sessionStorage.getItem('assistantOtherTeamsModel');
        populateAdvisorModelSelect(
            assistantAgentModelSelect,
            models,
            storedAgentModel || data.defaults?.agent_model
        );
        populateAdvisorModelSelect(
            assistantOtherModelSelect,
            models,
            storedOtherModel || data.defaults?.other_teams_model
        );
        advisorModelsLoaded = true;
        if (askAssistantBtn) askAssistantBtn.disabled = models.length === 0;
    } catch (error) {
        console.error('Error loading advisor models:', error);
        if (askAssistantBtn) askAssistantBtn.disabled = true;
    }
}

function loadAssistantPreferences() {
    const autoEnabled = sessionStorage.getItem('assistantAutoEnabled');
    const scope = sessionStorage.getItem('assistantScope');
    const simMode = sessionStorage.getItem('simEngineMode');
    if (assistantAutoToggle && autoEnabled !== null) {
        assistantAutoToggle.checked = autoEnabled === 'true';
    }
    if (assistantScopeSelect && scope) {
        assistantScopeSelect.value = scope;
    }
    if (simModeSelect && simMode) {
        simModeSelect.value = simMode;
    }
}

function usePolicySimMode() {
    return simModeSelect && simModeSelect.value === 'policy';
}

function buildSimRequestBody() {
    return JSON.stringify({ use_policy: usePolicySimMode() });
}

function getAssistantScope() {
    return assistantScopeSelect ? assistantScopeSelect.value : 'agent_only';
}

function getSnakeTeamOnTurn(state) {
    if (state.snake_team_on_turn != null) {
        return state.snake_team_on_turn;
    }
    const draftOrder = state.draft_order || [];
    const snakePickIdx = state.current_pick_number - 1;
    if (snakePickIdx >= 0 && snakePickIdx < draftOrder.length) {
        return draftOrder[snakePickIdx];
    }
    return null;
}

function isOverrideActive(state) {
    if (typeof state.override_active === 'boolean') {
        return state.override_active;
    }
    const snakeTeam = getSnakeTeamOnTurn(state);
    return snakeTeam != null && state.current_team_picking != null &&
        Number(state.current_team_picking) !== Number(snakeTeam);
}

function isDraftActive(state) {
    return Boolean(state && state.current_team_picking != null);
}

function getAdvisingTeamId(state, trigger) {
    if (trigger === 'auto') {
        return getSnakeTeamOnTurn(state);
    }
    return state.current_team_picking;
}

function shouldAutoAdvisor(state) {
    if (!assistantAutoToggle || !assistantAutoToggle.checked) return false;
    if (!isDraftActive(state)) return false;
    if (isOverrideActive(state)) return false;
    const snakeTeam = getSnakeTeamOnTurn(state);
    if (snakeTeam == null) return false;
    const agentTeamId = state.agent_start_position;
    const scope = getAssistantScope();
    if (scope === 'agent_only' && Number(snakeTeam) !== Number(agentTeamId)) {
        return false;
    }
    if (lastAutoAdvisorPickNumber === state.current_pick_number) return false;
    if (advisorInFlight) return false;
    return true;
}

function updateAssistantBanner(state) {
    if (!assistantBanner) return;
    if (!state || !isDraftActive(state)) {
        assistantBanner.classList.remove('visible');
        assistantBanner.textContent = '';
        return;
    }
    if (isOverrideActive(state)) {
        assistantBanner.textContent =
            `Clock overridden — Ask Assistant will advise Team ${state.current_team_picking}. Auto assistant paused.`;
        assistantBanner.classList.add('visible');
        return;
    }
    assistantBanner.classList.remove('visible');
    assistantBanner.textContent = '';
}

function updateAskAssistantButton(state) {
    if (!askAssistantBtn) return;
    askAssistantBtn.disabled = !isDraftActive(state);
}

function renderAdvisorLoading() {
    if (!advisorPanel || !advisorPanelBody) return;
    advisorPanel.classList.add('open');
    advisorPanelTitle.textContent = 'Assistant';
    advisorPanelBody.innerHTML = '<div class="advisor-loading">Thinking...</div>';
}

function renderAdvisorResult(data) {
    if (!advisorPanel || !advisorPanelBody) return;
    advisorPanel.classList.add('open');
    if (data.degraded) {
        renderAdvisorDegraded(data);
        return;
    }
    advisorPanelTitle.textContent = `Pick: ${data.recommended_name}`;
    const bullets = (data.rationale_bullets || [])
        .map(item => `<li>${item}</li>`)
        .join('');
    const alternates = (data.alternates || [])
        .map(item => `<div class="advisor-alternate"><strong>${item.name}</strong>: ${item.reason}</div>`)
        .join('');
    const flags = (data.flags || []).filter(item => item && item !== 'none');
    const unknowns = (data.unknown_factors || []).filter(Boolean);
    const metaParts = [];
    if (flags.length) metaParts.push(`Flags: ${flags.join(', ')}`);
    if (unknowns.length) metaParts.push(`Unknown: ${unknowns.join(', ')}`);
    advisorPanelBody.innerHTML = `
        <div><strong>Team ${data.advising_team_id}</strong> · confidence: ${data.confidence || 'n/a'}</div>
        ${bullets ? `<ul>${bullets}</ul>` : '<div class="advisor-muted">No rationale bullets returned — try Pro or Gemini if this persists.</div>'}
        ${alternates ? `<div><strong>Alternates</strong>${alternates}</div>` : ''}
        ${metaParts.length ? `<div class="advisor-meta">${metaParts.join(' · ')}</div>` : ''}
    `;
}

function renderAdvisorDegraded(data) {
    if (!advisorPanel || !advisorPanelBody) return;
    const pickLabel = data.recommended_name
        ? `Unverified: ${data.recommended_name}`
        : 'Unverified response';
    advisorPanelTitle.textContent = pickLabel;
    const bullets = (data.rationale_bullets || [])
        .map(item => `<li>${item}</li>`)
        .join('');
    const alternates = (data.alternates || [])
        .map(item => `<div class="advisor-alternate"><strong>${item.name}</strong>: ${item.reason}</div>`)
        .join('');
    const rawBlock = data.raw_content
        ? `<details><summary>Raw model output</summary><pre class="advisor-raw">${escapeHtml(data.raw_content)}</pre></details>`
        : '';
    advisorPanelBody.innerHTML = `
        <div class="advisor-degraded-warning">
            Could not fully validate this response. Review before drafting.
            ${data.parse_error ? `<br><span class="advisor-muted">${escapeHtml(data.parse_error)}</span>` : ''}
        </div>
        <div><strong>Team ${data.advising_team_id}</strong>${data.confidence ? ` · confidence: ${data.confidence}` : ''}</div>
        ${data.recommended_name ? `<div><strong>Suggested:</strong> ${escapeHtml(data.recommended_name)}${data.recommended_player_id ? ` (id ${data.recommended_player_id})` : ''}</div>` : ''}
        ${bullets ? `<ul>${bullets}</ul>` : ''}
        ${alternates ? `<div><strong>Alternates</strong>${alternates}</div>` : ''}
        ${rawBlock}
    `;
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function renderAdvisorError(message) {
    if (!advisorPanel || !advisorPanelBody) return;
    advisorPanel.classList.add('open');
    advisorPanelTitle.textContent = 'Assistant';
    advisorPanelBody.innerHTML = `<div class="advisor-error">${message}</div>`;
}

async function fetchAdvisor({ trigger = 'manual' } = {}) {
    if (!currentDraftState || advisorInFlight) return;
    const teamId = getAdvisingTeamId(currentDraftState, trigger);
    if (teamId == null) {
        renderAdvisorError('No team is currently on the clock.');
        return;
    }
    advisorInFlight = true;
    renderAdvisorLoading();
    const gpMinStr = gpFracMinInput ? gpFracMinInput.value : '';
    const payload = {
        team_id: teamId,
        scope: getAssistantScope(),
        trigger,
    };
    if (assistantAgentModelSelect && assistantAgentModelSelect.value) {
        payload.agent_model = assistantAgentModelSelect.value;
    }
    if (assistantOtherModelSelect && assistantOtherModelSelect.value) {
        payload.other_teams_model = assistantOtherModelSelect.value;
    }
    if (gpMinStr !== '' && !Number.isNaN(parseFloat(gpMinStr))) {
        payload.gp_min = parseFloat(gpMinStr);
    }
    try {
        const response = await fetch('/api/draft/advisor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `HTTP error! status: ${response.status}`);
        }
        renderAdvisorResult(data);
        if (trigger === 'auto') {
            lastAutoAdvisorPickNumber = currentDraftState.current_pick_number;
        }
    } catch (error) {
        console.error('Error fetching assistant recommendation:', error);
        renderAdvisorError(error.message || 'Assistant request failed.');
    } finally {
        advisorInFlight = false;
    }
}

function maybeAutoAdvisor(state) {
    if (!shouldAutoAdvisor(state)) return;
    window.setTimeout(() => {
        if (shouldAutoAdvisor(currentDraftState)) {
            fetchAdvisor({ trigger: 'auto' });
        }
    }, 150);
}

function setupSortHandlers() {
        const ths = document.querySelectorAll('#player-list th');
    ths.forEach(th => {
        const key = th.dataset.sortKey;
        if (!key) return;
        th.classList.add('sortable');
        if (!th.dataset.label) th.dataset.label = th.innerText;
        th.addEventListener('click', () => {
            if (currentSortBy === key) {
                currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
            } else {
                currentSortBy = key;
                    // Defaults: ADP/name/position/team/bye ascend; others descend
                    currentSortDir = (key === 'adp' || key === 'name' || key === 'position' || key === 'team' || key === 'bye_week') ? 'asc' : 'desc';
            }
            updateSortIndicators();
            fetchPlayers();
        });
    });
    updateSortIndicators();
}

function updateSortIndicators() {
    document.querySelectorAll('#player-list th').forEach(th => {
        const key = th.dataset.sortKey;
        if (!key) return;
        if (!th.dataset.label) th.dataset.label = th.textContent.replace(/ [▲▼]$/, '').trim();
        const arrow = (key === currentSortBy) ? (currentSortDir === 'asc' ? ' ▲' : ' ▼') : '';
        th.textContent = th.dataset.label + arrow;
    });
}

async function loadInitialState() {
    try {
        const response = await fetch('/api/draft/state');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        renderDraftState(data);
        fetchPlayers(); // Now fetch players
    } catch (error) {
        console.error('Error loading initial draft state:', error);
        alert(`Could not load draft state: ${error.message}`);
    }
}

const fetchPlayers = async () => {
    const searchInput = playerSearch.value;
    const selectedPositions = Array.from(positionCheckboxes)
                                   .filter(i => i.checked)
                                   .map(i => i.value);

    const params = new URLSearchParams();
    if (searchInput) params.append('search', searchInput);
    if (selectedPositions.length) params.append('position', selectedPositions.join(','));
    params.append('sort_by', currentSortBy);
    params.append('sort_dir', currentSortDir);

    try {
        const response = await fetch(`/api/players?${params.toString()}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        let players = await response.json();
        const gpMinStr = gpFracMinInput.value;
        if (gpMinStr !== '' && !Number.isNaN(parseFloat(gpMinStr))) {
            const gpMin = parseFloat(gpMinStr);
            players = players.filter(p => {
                const v = p.games_played_frac;
                if (v === 'R') return true;
                if (v === null || v === undefined) return false;
                const num = Number(v);
                if (!isFinite(num)) return false;
                return num >= gpMin;
            });
        }
        renderPlayers(players);
    } catch (error) {
        console.error('Error fetching players:', error);
        playerTbody.innerHTML = `<tr><td colspan="13" style="color: red;">Error loading players: ${error.message}</td></tr>`;
    }
};

async function updatePositionMetaChips() {
    try {
        const resp = await fetch('/api/players');
        if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
        const allPlayers = await resp.json();
        const positions = ['QB', 'RB', 'WR', 'TE'];
        const byPos = { QB: [], RB: [], WR: [], TE: [] };
        allPlayers.forEach(p => { if (byPos[p.position]) byPos[p.position].push(p); });

        function stats(arr) {
            if (!arr || arr.length === 0) return { minAdp: 'N/A', maxVorp: 'N/A', maxPts: 'N/A' };
            const adps = arr.map(p => p.adp).filter(v => v !== null && v !== undefined && isFinite(Number(v))).map(Number);
            const vorps = arr.map(p => p.vorp).filter(v => v !== null && v !== undefined && isFinite(Number(v))).map(Number);
            const pts = arr.map(p => p.projected_points).filter(v => v !== null && v !== undefined && isFinite(Number(v))).map(Number);
            return {
                minAdp: adps.length ? Math.min(...adps).toFixed(1) : 'N/A',
                maxVorp: vorps.length ? Math.max(...vorps).toFixed(1) : 'N/A',
                maxPts: pts.length ? Math.max(...pts).toFixed(1) : 'N/A',
            };
        }

        const qbStats = stats(byPos.QB);
        const rbStats = stats(byPos.RB);
        const wrStats = stats(byPos.WR);
        const teStats = stats(byPos.TE);

        const allStats = stats(allPlayers);
        const setChipValues = (chipId, s) => {
            const chip = document.getElementById(chipId);
            if (!chip) return;
            const adpEl = chip.querySelector('[data-field="adp"], #overall-min-adp');
            const vorpEl = chip.querySelector('[data-field="vorp"], #overall-max-vorp');
            const ptsEl = chip.querySelector('[data-field="pts"], #overall-max-pts');
            if (adpEl) adpEl.textContent = s.minAdp;
            if (vorpEl) vorpEl.textContent = s.maxVorp;
            if (ptsEl) ptsEl.textContent = s.maxPts;
        };

        setChipValues('overall-avail-chip', allStats);
        setChipValues('qb-avail-chip', qbStats);
        setChipValues('rb-avail-chip', rbStats);
        setChipValues('wr-avail-chip', wrStats);
        setChipValues('te-avail-chip', teStats);
    } catch (e) {
        console.error('Error updating position meta chips:', e);
    }
}

// ----- Coloring Utilities -----
const POS_COLORS = {
    'QB': '#f8d7da',  // light red
    'RB': '#cfe2ff',  // light blue
    'WR': '#e2f7cf',  // light green
    'TE': '#ffe8cc',  // light orange
};

// *** NEW: Store official team colors ***
const TEAM_COLORS = {
    "BAL": { background: "#655C9D", text: "#FFFFFF" },
    "CIN": { background: "#FC835A", text: "#000000" },
    "CLE": { background: "#6E604C", text: "#FF3C00" },
    "PIT": { background: "#FFCB59", text: "#101820" },
    "BUF": { background: "#4C70AF", text: "#C60C30" },
    "MIA": { background: "#4CAFB6", text: "#FC4C02" },
    "NE": { background: "#4C647C", text: "#C60C30" },
    "NYJ": { background: "#598979", text: "#FFFFFF" },
    "HOU": { background: "#4E626D", text: "#A71930" },
    "IND": { background: "#4C6B8F", text: "#A2AAAD" },
    "JAX": { background: "#575D62", text: "#D7A22A" },
    "TEN": { background: "#546579", text: "#4B92DB" },
    "DEN": { background: "#FC835A", text: "#002244" },
    "KC": { background: "#EB5D73", text: "#FFB81C" },
    "LV": { background: "#4C4C4C", text: "#A5ACAF" },
    "LAC": { background: "#4CA6D7", text: "#FFC20E" },
    "CHI": { background: "#545B69", text: "#C83803" },
    "DET": { background: "#4C9FCB", text: "#B0B7BC" },
    "GB": { background: "#62736E", text: "#FFB612" },
    "MIN": { background: "#8367A8", text: "#FFC62F" },
    "DAL": { background: "#4C71B4", text: "#FFFFFF" },
    "NYG": { background: "#546493", text: "#A71930" },
    "PHI": { background: "#4C8187", text: "#A5ACAF" },
    "WAS": { background: "#8B5A5A", text: "#FFB612" },
    "ATL": { background: "#C15E6E", text: "#000000" },
    "CAR": { background: "#4CA9D9", text: "#101820" },
    "NO": { background: "#E0D0AF", text: "#101820" },
    "TB": { background: "#E15353", text: "#FF7900" },
    "ARI": { background: "#B66578", text: "#FFFFFF" },
    "LA": { background: "#4C71B4", text: "#FFA300" },
    "SF": { background: "#C34C4C", text: "#B3995D" },
    "SEA": { background: "#4C647C", text: "#69BE28" }
};

// *** UPDATED: New function to look up team colors ***
function getTeamColor(team) {
    // Returns an object { background, text } or null if not found
    if (!team || !TEAM_COLORS[team]) return null;
    return TEAM_COLORS[team];
}


function clamp(val, min, max) { return Math.max(min, Math.min(max, val)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function lerpColor(colorA, colorB, t) {
    // colors as [r,g,b]
    const r = Math.round(lerp(colorA[0], colorB[0], t));
    const g = Math.round(lerp(colorA[1], colorB[1], t));
    const b = Math.round(lerp(colorA[2], colorB[2], t));
    return `rgb(${r}, ${g}, ${b})`;
}
const GREEN = [200, 245, 200]; // light green
const RED = [245, 200, 200];   // light red
// Dynamic, multi-stop color scale: green (good) -> yellow -> orange -> red (bad)
function _computeMetricColor(value, minVal, maxVal, tMapper) {
    if (
        value === null || value === undefined || Number.isNaN(value) ||
        minVal === null || maxVal === null || maxVal === minVal
    ) return '';

    const tRaw = (value - minVal) / (maxVal - minVal);
    const t = clamp(tMapper(tRaw), 0, 1);

    const RED = [245, 200, 200];
    const ORANGE = [255, 218, 185];
    const YELLOW = [255, 255, 224];
    const GREEN = [200, 245, 200];

    // Map t in [0,1] from bad->good: 0=RED, ~0.33=ORANGE, ~0.66=YELLOW, 1=GREEN
    if (t < 1/3) {
        return lerpColor(RED, ORANGE, t * 3);
    } else if (t < 2/3) {
        return lerpColor(ORANGE, YELLOW, (t - 1/3) * 3);
    } else {
        return lerpColor(YELLOW, GREEN, (t - 2/3) * 3);
    }
}

function getStandardMetricColor(value, minVal, maxVal) {
    return _computeMetricColor(value, minVal, maxVal, t => t);
}

function getInvertedMetricColor(value, minVal, maxVal) {
    return _computeMetricColor(value, minVal, maxVal, t => 1 - t);
}

function openPlayerInfo(playerName) {
    const query = `${playerName} 2026 outlook`;
    const url = 'https://www.google.com/search?q=' + encodeURIComponent(query);
    window.open(url, '_blank');
}

const DEPTH_ROLE_LABELS = {
    starter: 'Starter',
    co_starter: 'Co',
    committee: 'Cmte',
    backup: 'Bkup',
    unknown: '—',
};

function formatDepthRoleLabel(depthRole) {
    if (!depthRole || depthRole === 'unknown') return '—';
    return DEPTH_ROLE_LABELS[depthRole] || formatInsightTagLabel(depthRole);
}

function formatPlayingTimeLabel(tier) {
    if (!tier || tier === 'unknown') return null;
    return `Playing time: ${tier}`;
}

function showOutlookTooltip(player, event) {
    const insight = player.insight;
    if (!insight || !insight.outlook_phrase) return;
    outlookTooltipPhrase.textContent = insight.outlook_phrase;
    const roleParts = [
        formatDepthRoleLabel(insight.depth_role),
        formatPlayingTimeLabel(insight.playing_time_tier),
    ].filter(part => part && part !== '—');
    outlookTooltipRole.textContent = roleParts.length
        ? roleParts.join(' · ')
        : '';
    outlookTooltipRole.style.display = roleParts.length ? 'block' : 'none';
    outlookTooltipSummary.textContent = insight.summary || '';
    outlookTooltip.style.display = 'block';
    positionOutlookTooltip(event);
}

function positionOutlookTooltip(event) {
    const padding = 12;
    const rect = outlookTooltip.getBoundingClientRect();
    let left = event.clientX + padding;
    let top = event.clientY + padding;
    if (left + rect.width > window.innerWidth - padding) {
        left = event.clientX - rect.width - padding;
    }
    if (top + rect.height > window.innerHeight - padding) {
        top = event.clientY - rect.height - padding;
    }
    outlookTooltip.style.left = `${Math.max(padding, left)}px`;
    outlookTooltip.style.top = `${Math.max(padding, top)}px`;
}

function hideOutlookTooltip() {
    outlookTooltip.style.display = 'none';
}

function formatInsightTagLabel(value) {
    if (!value) return '';
    return String(value).replace(/_/g, ' ');
}

function openInsightModal(player) {
    const insight = player.insight;
    if (!insight) return;
    modalPlayerName = player.name;
    insightModalTitle.textContent = player.name;
    insightModalPhrase.textContent = insight.outlook_phrase || '';
    insightModalSummary.textContent = insight.summary || '';
    insightModalTags.innerHTML = '';
    const tagValues = [
        ...(insight.tags || []),
        insight.depth_role,
        insight.injury_risk ? `injury: ${insight.injury_risk}` : null,
    ].filter(Boolean);
    tagValues.forEach(tag => {
        const span = document.createElement('span');
        span.className = 'insight-tag';
        span.textContent = formatInsightTagLabel(tag);
        insightModalTags.appendChild(span);
    });
    insightModalBullets.innerHTML = '';
    (insight.bullets || []).forEach(bullet => {
        const li = document.createElement('li');
        const link = document.createElement('a');
        link.href = bullet.source_url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = bullet.text;
        li.appendChild(link);
        if (bullet.source_domain) {
            const source = document.createElement('span');
            source.textContent = ` (${bullet.source_domain})`;
            source.style.color = '#888';
            source.style.fontSize = '0.8rem';
            li.appendChild(source);
        }
        insightModalBullets.appendChild(li);
    });
    insightModalConfidence.textContent = insight.overall_confidence
        ? `Confidence: ${insight.overall_confidence}`
        : '';
    insightModal.classList.add('open');
}

function closeInsightModal() {
    insightModal.classList.remove('open');
    modalPlayerName = null;
}

insightModalClose.addEventListener('click', closeInsightModal);
insightModalGoogle.addEventListener('click', () => {
    if (modalPlayerName) openPlayerInfo(modalPlayerName);
});
insightModal.addEventListener('click', (e) => {
    if (e.target === insightModal) closeInsightModal();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && insightModal.classList.contains('open')) {
        closeInsightModal();
    }
});

async function loadInsightsMeta() {
    try {
        const response = await fetch('/api/insights/meta');
        if (!response.ok) return;
        const meta = await response.json();
        if (!meta.available) {
            insightsHint.textContent = 'No player insights file found';
            insightsHint.classList.add('missing');
            return;
        }
        const generated = meta.generated_at
            ? new Date(meta.generated_at).toLocaleString()
            : 'unknown date';
        const hint = `${meta.enriched_player_count} players enriched (${generated})`;
        insightsHint.textContent = hint;
        insightsHint.classList.remove('missing');
        if (outlookColHeader) {
            outlookColHeader.title = `Outlook from ${meta.source_file || 'insights export'} — ${hint}`;
        }
    } catch (error) {
        console.warn('Could not load insights meta:', error);
    }
}

function formatSleeperStatusDisplay(injuryStatus, rosterStatus, depthChart) {
    const hasRosterStatus = Boolean(rosterStatus);
    const statusPart = injuryStatus
        || (rosterStatus && rosterStatus !== 'Active' ? rosterStatus : null);
    const depthPart = depthChart || null;
    const parts = [statusPart, depthPart].filter(Boolean);
    const text = parts.length
        ? parts.join(' · ')
        : (hasRosterStatus ? '' : 'N/A');

    const titleParts = [];
    if (rosterStatus) titleParts.push(`Sleeper status: ${rosterStatus}`);
    if (injuryStatus) titleParts.push(`Injury: ${injuryStatus}`);
    if (depthPart) titleParts.push(`Depth: ${depthPart}`);
    const title = titleParts.length ? titleParts.join(' · ') : 'No Sleeper match';

    const style = {};
    if (injuryStatus) {
        style.backgroundColor = '#ffdada';
        style.color = '#7a1f1f';
        style.fontWeight = 'bold';
    } else if (rosterStatus && rosterStatus !== 'Active') {
        style.backgroundColor = '#fff3cd';
        style.color = '#6b5300';
    }
    return { text, title, style };
}

const renderPlayers = (players) => {
    lastRenderedPlayers = players;
    playerTbody.innerHTML = '';
    if (players.length === 0) {
        const row = playerTbody.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 13;
        cell.innerText = 'No players found with current filters.';
        return;
    }

    // ----- Compute dynamic ranges from currently visible players -----
    const numberOrNull = (v) => (v === null || v === undefined || Number.isNaN(Number(v))) ? null : Number(v);
    const finiteNumber = (v) => (typeof v === 'number' && isFinite(v));

    const adpValues = players
        .map(p => numberOrNull(p.adp))
        .filter(v => v !== null && isFinite(v));
    const ptsValues = players
        .map(p => numberOrNull(p.projected_points))
        .filter(v => v !== null && isFinite(v));
    const vorpValues = players
        .map(p => numberOrNull(p.vorp))
        .filter(v => v !== null && isFinite(v));
    const gpValues = players
        .map(p => (p.games_played_frac === 'R' ? null : numberOrNull(p.games_played_frac)))
        .filter(v => v !== null && isFinite(v));

    const minMax = (arr) => arr.length ? [Math.min(...arr), Math.max(...arr)] : [null, null];
    const [adpMin, adpMax] = minMax(adpValues);
    const [ptsMin, ptsMax] = minMax(ptsValues);
    const [vorpMin, vorpMax] = minMax(vorpValues);
    const [gpMin, gpMax] = minMax(gpValues);

    players.forEach(player => {
        const row = playerTbody.insertRow();
        row.dataset.playerId = player.player_id;
        // Blind toggle cell
        const blindCell = row.insertCell();
        const blindCheckbox = document.createElement('input');
        blindCheckbox.type = 'checkbox';
        blindCheckbox.title = 'Exclude from AI suggestion';
        blindCheckbox.checked = blindSet.has(player.player_id);
        blindCheckbox.addEventListener('click', (e) => e.stopPropagation());
        blindCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) blindSet.add(player.player_id); else blindSet.delete(player.player_id);
            if (currentDraftState) updateHeaderSuggestion(currentDraftState);
        });
        blindCell.appendChild(blindCheckbox);
        // ADP
        const adpCell = row.insertCell();
        const adpValNum = (player.adp !== null && player.adp !== undefined) ? Number(player.adp) : null;
        adpCell.innerText = (adpValNum !== null) ? adpValNum.toFixed(1) : 'N/A';
        // Name
        const nameCell = row.insertCell();
        nameCell.innerText = player.name;
        // Role
        const roleCell = row.insertCell();
        roleCell.classList.add('role-col');
        if (player.insight && player.insight.depth_role) {
            const depthRole = player.insight.depth_role;
            roleCell.innerText = formatDepthRoleLabel(depthRole);
            if (depthRole !== 'unknown') {
                roleCell.classList.add(`role-${depthRole}`);
                roleCell.title = formatInsightTagLabel(depthRole);
            } else {
                roleCell.classList.add('missing');
                roleCell.title = 'Role unknown';
            }
        } else {
            roleCell.innerText = '—';
            roleCell.classList.add('missing');
            roleCell.title = 'No offline insight for this player';
        }
        // Outlook
        const outlookCell = row.insertCell();
        outlookCell.classList.add('outlook-col');
        if (player.insight && player.insight.outlook_phrase) {
            outlookCell.innerText = player.insight.outlook_phrase;
            outlookCell.title = player.insight.outlook_phrase;
            outlookCell.classList.add('has-insight');
            outlookCell.addEventListener('mouseenter', (e) => {
                showOutlookTooltip(player, e);
            });
            outlookCell.addEventListener('mousemove', (e) => {
                if (outlookTooltip.style.display === 'block') positionOutlookTooltip(e);
            });
            outlookCell.addEventListener('mouseleave', hideOutlookTooltip);
        } else {
            outlookCell.innerText = '—';
            outlookCell.classList.add('missing');
            outlookCell.title = 'No offline insight for this player';
        }
        // Position
        const posCell = row.insertCell();
        posCell.innerText = player.position;
        // Projected Points
        const ptsCell = row.insertCell();
        const ptsVal = Number(player.projected_points) || 0;
        ptsCell.innerText = ptsVal.toFixed(1);
        // VORP
        const vorpCell = row.insertCell();
        const vorpVal = Number(player.vorp) || 0;
        vorpCell.innerText = vorpVal.toFixed(1);

        // Games Played Fraction
        const gpCell = row.insertCell();
        const gpRawVal = player.games_played_frac;
        const isRookie = (gpRawVal === 'R');
        if (isRookie) {
            gpCell.innerText = 'R';
            gpCell.style.backgroundColor = '#e0ccff';
            gpCell.style.color = '#3b2a64';
            gpCell.style.fontWeight = 'bold';
        } else {
            const gpValNum = (gpRawVal !== null && gpRawVal !== undefined && !Number.isNaN(Number(gpRawVal))) ? Number(gpRawVal) : null;
            gpCell.innerText = (gpValNum !== null) ? gpValNum.toFixed(2) : 'N/A';
            gpCell.style.color = '';
            gpCell.style.fontWeight = '';
            if (gpValNum !== null) {
                gpCell.style.background = getStandardMetricColor(gpValNum, gpMin, gpMax);
            } else {
                gpCell.style.background = '';
            }
        }

        // Team
        const teamCell = row.insertCell();
        const teamVal = player.team || 'N/A';
        teamCell.innerText = teamVal;
        // Bye
        const byeCell = row.insertCell();
        byeCell.innerText = player.bye_week;

        // Sleeper Status / Injury
        const statusCell = row.insertCell();
        const statusDisplay = formatSleeperStatusDisplay(
            player.sleeper_injury_status,
            player.sleeper_status,
            player.sleeper_depth_chart_position,
        );
        statusCell.innerText = statusDisplay.text;
        statusCell.title = statusDisplay.title;
        Object.assign(statusCell.style, statusDisplay.style);

        // Info buttons
        const infoCell = row.insertCell();
        infoCell.classList.add('info-col');
        const infoActions = document.createElement('div');
        infoActions.className = 'info-actions';
        if (player.insight) {
            const detailBtn = document.createElement('button');
            detailBtn.className = 'info-button';
            detailBtn.title = 'View player insight';
            detailBtn.textContent = 'i';
            detailBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                openInsightModal(player);
            });
            const googleBtn = document.createElement('button');
            googleBtn.className = 'info-button';
            googleBtn.title = 'Open Google search';
            googleBtn.textContent = '↗';
            googleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                openPlayerInfo(player.name);
            });
            infoActions.appendChild(detailBtn);
            infoActions.appendChild(googleBtn);
        } else {
            const googleBtn = document.createElement('button');
            googleBtn.className = 'info-button';
            googleBtn.title = 'Open Google search';
            googleBtn.textContent = '↗';
            googleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                openPlayerInfo(player.name);
            });
            infoActions.appendChild(googleBtn);
        }
        infoCell.appendChild(infoActions);

        // ----- Conditional Coloring -----
        if (adpValNum !== null) adpCell.style.background = getInvertedMetricColor(adpValNum, adpMin, adpMax);
        ptsCell.style.background = getStandardMetricColor(ptsVal, ptsMin, ptsMax);
        vorpCell.style.background = getStandardMetricColor(vorpVal, vorpMin, vorpMax);

        const posColor = POS_COLORS[player.position];
        if (posColor) {
            posCell.style.backgroundColor = posColor;
        }

        const teamColors = getTeamColor(teamVal);
        if (teamColors) {
            teamCell.style.backgroundColor = teamColors.background;
            teamCell.style.color = teamColors.text;
            byeCell.style.backgroundColor = teamColors.background;
            byeCell.style.color = teamColors.text;
        }

        row.addEventListener('click', () => draftPlayer(player.player_id));
    });
};

// Event Listeners
playerSearch.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') return;
    fetchPlayers();
});
playerSearch.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    if (!lastRenderedPlayers.length) return;
    const topPlayer = lastRenderedPlayers[0];
    playerSearch.value = '';
    await draftPlayer(topPlayer.player_id);
});
positionCheckboxes.forEach(checkbox => checkbox.addEventListener('change', fetchPlayers));
gpFracMinInput.addEventListener('input', fetchPlayers);
if (newDraftBtn) newDraftBtn.addEventListener('click', startNewDraft);
undoPickBtn.addEventListener('click', undoLastPick);
simulatePickBtn.addEventListener('click', simulateNextPick);
autoDraftBtn.addEventListener('click', autoDraftRest);
downloadCsvBtn.addEventListener('click', downloadCsv);
if (simulateSeasonBtn) simulateSeasonBtn.addEventListener('click', simulateSeason);

document.addEventListener('DOMContentLoaded', () => {
    initBottomTabs();
    setupSortHandlers();
    loadAssistantPreferences();
    loadAdvisorModels();
    loadInsightsMeta();
    loadInitialState();
    updatePositionMetaChips();
    if (assistantAutoToggle) {
        assistantAutoToggle.addEventListener('change', () => {
            sessionStorage.setItem('assistantAutoEnabled', String(assistantAutoToggle.checked));
        });
    }
    if (assistantScopeSelect) {
        assistantScopeSelect.addEventListener('change', () => {
            sessionStorage.setItem('assistantScope', assistantScopeSelect.value);
        });
    }
    if (assistantAgentModelSelect) {
        assistantAgentModelSelect.addEventListener('change', () => {
            sessionStorage.setItem('assistantAgentModel', assistantAgentModelSelect.value);
        });
    }
    if (assistantOtherModelSelect) {
        assistantOtherModelSelect.addEventListener('change', () => {
            sessionStorage.setItem('assistantOtherTeamsModel', assistantOtherModelSelect.value);
        });
    }
    if (simModeSelect) {
        simModeSelect.addEventListener('change', () => {
            sessionStorage.setItem('simEngineMode', simModeSelect.value);
        });
    }
    if (askAssistantBtn) {
        askAssistantBtn.addEventListener('click', () => fetchAdvisor({ trigger: 'manual' }));
    }
    if (advisorPanelClose) {
        advisorPanelClose.addEventListener('click', () => advisorPanel.classList.remove('open'));
    }
});

async function startNewDraft() {
    if (!confirm('Are you sure you want to start a new draft? This will archive the current draft state.')) {
        return;
    }
    try {
        const response = await fetch('/api/draft/new', { method: 'POST' });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        renderDraftState(data);
        fetchPlayers(); // Refresh player list
    } catch (error) {
        console.error('Error starting new draft:', error);
        alert(`Error starting new draft: ${error.message}`);
    }
}

function syncAutoDraftButtonState() {
    const sched = currentDraftState && Array.isArray(currentDraftState.draft_order) &&
        typeof currentDraftState.current_pick_index === 'number' &&
        currentDraftState.current_pick_index < currentDraftState.draft_order.length;
    autoDraftBtn.disabled = !sched;
}

function renderDraftState(state) {
    if (previousPickNumber != null && state.current_pick_number < previousPickNumber) {
        lastAutoAdvisorPickNumber = null;
    }
    previousPickNumber = state.current_pick_number;
    currentDraftState = state;
    currentPickDisplay.innerText = state.current_pick_number;
    teamOnClockDisplay.innerText = state.current_team_picking;

    updateHeaderSuggestion(state);

    fetchDraftSummary();
    updatePositionMetaChips();

    const numTeams = Number(state.num_teams) || 10;
    const agentTeamId = state.agent_start_position;
    const teamOnClock = state.current_team_picking;
    const draftOrder = state.draft_order || [];
    const totalRounds = state.total_roster_size_per_team || 16;

    const snakePickIdx = state.current_pick_number - 1;
    const snakePickTeam = (snakePickIdx >= 0 && snakePickIdx < draftOrder.length)
        ? draftOrder[snakePickIdx] : null;

    // --- Render Board Headers (also serve as team selection buttons) ---
    boardHeaders.innerHTML = '';
    for (let i = 1; i <= numTeams; i++) {
        const teamId = i;
        const roster = state.team_rosters[teamId];
        const pointsSummary = state.team_points_summary[teamId] || { starters_total: 0, bench_total: 0 };
        const totalProjectedPoints = state.team_projected_points[teamId] || 0;

        let totalPlayers = 0;
        if (roster) {
            const starters = roster.starters || {};
            const startersCount = Object.values(starters).reduce(
                (sum, arr) => sum + (Array.isArray(arr) ? arr.length : 0), 0);
            const benchCount = (roster.bench || []).length;
            totalPlayers = startersCount + benchCount;
        }

        const byeWeeksData = state.team_bye_weeks?.[teamId] || {};
        let byeWeekCellsHTML = '';
        for (let week = 4; week <= 14; week++) {
            const weekData = byeWeeksData[week];
            let totalOnBye = 0;
            let posParts = [];
            if (weekData) {
                totalOnBye = Object.values(weekData).reduce((s, c) => s + c, 0);
                posParts = Object.entries(weekData)
                    .filter(([, c]) => c > 0)
                    .map(([p, c]) => `${p}:${c}`);
            }
            const posContent = posParts.join(' ');
            const intensity = Math.min(totalOnBye / 3, 1);
            const bgColor = totalOnBye > 1 ? `rgba(255, 100, 100, ${intensity})` : '#efefef';
            byeWeekCellsHTML += `<div class="board-bye-week-cell" style="background-color: ${bgColor};" title="Week ${week}${posContent ? ': ' + posParts.join(', ') : ''}">
                <span class="bye-week-num">${week}</span>
                ${posContent ? `<span class="bye-week-pos">${posContent}</span>` : ''}
            </div>`;
        }

        const th = document.createElement('th');
        const headerDiv = document.createElement('div');
        headerDiv.className = 'board-team-header board-team-header-btn';
        headerDiv.title = 'Click to set this team as picking';
        headerDiv.setAttribute('role', 'button');
        headerDiv.tabIndex = 0;

        if (teamId === agentTeamId) headerDiv.classList.add('main-team');
        if (teamId === snakePickTeam) headerDiv.classList.add('snake-order-pick');
        if (teamId === teamOnClock) headerDiv.classList.add('picking-now');
        if (state.team_is_full && state.team_is_full[teamId]) headerDiv.classList.add('team-full');

        headerDiv.innerHTML = `
            <span class="board-team-name">Team ${teamId}</span>
            <div class="board-team-stats">
                <span class="stat" title="Total players drafted">P: ${totalPlayers}</span>
                <span class="stat" title="Total projected points">${totalProjectedPoints.toFixed(0)} pts</span>
                <span class="stat" title="Starters points">S: ${pointsSummary.starters_total.toFixed(0)}</span>
                <span class="stat" title="Bench points">B: ${pointsSummary.bench_total.toFixed(0)}</span>
            </div>
            <div class="board-bye-week-grid" title="Bye week distribution (weeks 4-14)">
                ${byeWeekCellsHTML}
            </div>
        `;

        headerDiv.addEventListener('click', () => overrideTeam(teamId));
        headerDiv.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                overrideTeam(teamId);
            }
        });

        th.appendChild(headerDiv);
        boardHeaders.appendChild(th);
    }

    // --- Render Board Rows (Rounds) ---
    boardRows.innerHTML = '';
    for (let r = 0; r < totalRounds; r++) {
        const tr = document.createElement('tr');
        for (let t = 1; t <= numTeams; t++) {
            const teamId = t;
            const td = document.createElement('td');
            const roster = state.team_rosters[teamId] || {};
            const players = roster.players_flat || [];
            const player = players[r];

            const cell = document.createElement('div');
            cell.className = 'board-cell';

            // Check if this cell is currently on clock
            // Pick index = r * numTeams + (r % 2 === 0 ? t-1 : numTeams - t)
            // But we want to know if it's Team t's turn in Round r.
            // If team t is on clock and they have r players, then Round r+1 cell is on clock.
            const isNextForTeam = (teamId === teamOnClock && players.length === r);
            if (isNextForTeam) cell.classList.add('on-clock');

            if (player) {
                cell.style.backgroundColor = POS_COLORS[player.position] || '#fff';
                const teamColors = getTeamColor(player.team);
                if (teamColors) {
                    // Subtle indicator for NFL team
                    cell.style.borderLeft = `4px solid ${teamColors.background}`;
                }

                cell.innerHTML = `
                    <span class="pos">${player.position}</span>
                    <span class="name">${player.name}</span>
                    <span class="team">${player.team || 'N/A'} (B:${player.bye_week})</span>
                `;
                cell.title = 'Click to transfer this player';
                cell.addEventListener('click', () => transferDraftedPlayer(player, teamId));
            } else {
                cell.classList.add('empty');
                cell.innerText = `Round ${r + 1}`;
            }
            td.appendChild(cell);
            tr.appendChild(td);
        }
        boardRows.appendChild(tr);
    }

    syncAutoDraftButtonState();
    updateAssistantBanner(state);
    updateAskAssistantButton(state);
    maybeAutoAdvisor(state);
}

async function fetchAiSuggestionForTeam(teamId, ignoreIds) {
    const params = new URLSearchParams({ team_id: String(teamId) });
    if (ignoreIds && ignoreIds.size > 0) params.append('ignore', Array.from(ignoreIds).join(','));
    const response = await fetch(`/api/draft/ai_suggestion_for_team?${params.toString()}`);
    return await response.json();
}

async function updateHeaderSuggestion(state) {
    const teamOnClock = state.current_team_picking;
    if (!teamOnClock) {
        aiSuggestionDisplay.innerHTML = 'N/A';
        return;
    }
    if (aiChip) aiChip.classList.add('loading');
    try {
        const data = await fetchAiSuggestionForTeam(teamOnClock, blindSet);
        if (data && !data.error) {
            const POS_ORDER = ['QB', 'RB', 'WR', 'TE'];
            const suggestionHTML = POS_ORDER.map(pos => {
                const prob = data[pos] || 0;
                return `<div class="ai-suggestion-item">
                            <span class="ai-pos">${pos}</span>
                            <span class="ai-prob">${Math.round(prob * 100)}%</span>
                        </div>`;
            }).join('');
            aiSuggestionDisplay.innerHTML = suggestionHTML || 'N/A';
        } else {
            aiSuggestionDisplay.innerText = data.error || 'N/A';
        }
    } catch (e) {
        console.error('Error fetching AI suggestion for team:', e);
        aiSuggestionDisplay.innerText = 'Error';
    } finally {
        if (aiChip) aiChip.classList.remove('loading');
    }
}

// Removed bulk prefetch; suggestions are fetched on demand for the team on clock

async function fetchDraftSummary() {
    try {
        const response = await fetch('/api/draft/summary');
        const data = await response.json();
        if (response.ok) {
            totalPicksDisplay.innerText = data.total_picks;
            qbDraftedDisplay.innerText = data.picks_by_position.QB;
            rbDraftedDisplay.innerText = data.picks_by_position.RB;
            wrDraftedDisplay.innerText = data.picks_by_position.WR;
            teDraftedDisplay.innerText = data.picks_by_position.TE;
        } else {
            totalPicksDisplay.innerText = 'Error';
        }
    } catch (error) {
        console.error('Error fetching draft summary:', error);
        totalPicksDisplay.innerText = 'Error';
    }
}

async function draftPlayer(playerId) {
    if (!currentDraftState) {
        alert("Please start a new draft before selecting a player.");
        return;
    }

    try {
        const response = await fetch('/api/draft/pick', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_id: playerId })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        
        renderDraftState(data);
        fetchPlayers();
    } catch (error) {
        console.error('Error drafting player:', error);
        alert(`Error drafting player: ${error.message}`);
    }
}

async function undoLastPick() {
    if (!currentDraftState) {
        alert("No draft in progress to undo.");
        return;
    }
    try {
        const response = await fetch('/api/draft/undo', { method: 'POST' });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        renderDraftState(data);
        fetchPlayers();
    } catch (error) {
        console.error('Error undoing pick:', error);
        alert(`Error undoing pick: ${error.message}`);
    }
}

async function overrideTeam(teamId) {
    if (!currentDraftState) {
        alert("Please start a new draft first.");
        return;
    }
    try {
        const response = await fetch('/api/draft/override_team', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team_id: teamId })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        renderDraftState(data);
    } catch (error) {
        console.error('Error overriding team:', error);
        alert(`Error overriding team: ${error.message}`);
    }
}

async function transferDraftedPlayer(player, fromTeamId) {
    if (!currentDraftState) {
        alert("Please start a new draft first.");
        return;
    }
    const destination = prompt(
        `Move ${player.name} from Team ${fromTeamId} to which team?`,
        String(fromTeamId)
    );
    if (destination === null) {
        return;
    }
    const toTeamId = Number(destination);
    if (!Number.isInteger(toTeamId)) {
        alert("Enter a valid team number.");
        return;
    }
    try {
        const response = await fetch('/api/draft/transfer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_id: player.player_id, to_team_id: toTeamId })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `HTTP error! status: ${response.status}`);
        }
        renderDraftState(data);
        fetchPlayers();
    } catch (error) {
        console.error('Error transferring player:', error);
        alert(`Error transferring player: ${error.message}`);
    }
}

async function simulateNextPick() {
    if (!currentDraftState) {
        alert("Please start a new draft first.");
        return;
    }
    
    try {
        const response = await fetch('/api/draft/simulate_pick', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: buildSimRequestBody(),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        renderDraftState(data);
        fetchPlayers();

    } catch (error) {
        console.error('Error simulating pick:', error);
        alert(`Error simulating pick: ${error.message}`);
    }
}

async function autoDraftRest() {
    if (!currentDraftState) {
        alert("Please start a new draft first.");
        return;
    }
    const message =
        "This will automatically simulate every remaining pick for computer-controlled teams. " +
        "You cannot undo this in one step. Continue?";
    if (!confirm(message)) {
        return;
    }

    autoDraftBtn.disabled = true;
    autoDraftBtn.textContent = "Drafting...";

    try {
        const response = await fetch("/api/draft/simulate_rest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: buildSimRequestBody(),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        renderDraftState(data);
        fetchPlayers();
    } catch (error) {
        console.error("Error auto-drafting:", error);
        alert(`Error auto-drafting: ${error.message}`);
    } finally {
        autoDraftBtn.textContent = AUTO_DRAFT_LABEL;
        syncAutoDraftButtonState();
    }
}

function downloadCsv() {
    window.location.href = '/api/draft/export_csv';
}

function formatTeamLabel(teamId, fallback = 'Unknown') {
    if (teamId != null && Number.isFinite(Number(teamId))) {
        return `Team ${teamId}`;
    }
    return fallback || 'Unknown';
}

function isAgentTeam(teamId) {
    const agentTeamId = currentDraftState && currentDraftState.agent_start_position;
    return agentTeamId != null && teamId != null && Number(teamId) === Number(agentTeamId);
}

function formatStandingsRank(rank) {
    if (rank === 1) return '<span class="rank-badge rank-1">1st</span>';
    if (rank === 2) return '<span class="rank-badge rank-2">2nd</span>';
    if (rank === 3) return '<span class="rank-badge rank-3">3rd</span>';
    return `<span class="rank-badge">${rank}</span>`;
}

function renderMatchupCard(awayTeam, homeTeam, awayScore, homeScore, awayTeamId, homeTeamId) {
    const awayLabel = awayTeam || 'BYE';
    const homeLabel = homeTeam || 'BYE';
    const awayValue = awayScore == null ? '—' : Number(awayScore).toFixed(2);
    const homeValue = homeScore == null ? '—' : Number(homeScore).toFixed(2);
    const awayWins = awayScore != null && homeScore != null && awayScore > homeScore;
    const homeWins = awayScore != null && homeScore != null && homeScore > awayScore;
    const isTie = awayScore != null && homeScore != null && awayScore === homeScore;

    function buildRow(teamLabel, scoreValue, teamId, isWinner, isLoser) {
        const classes = ['matchup-row'];
        if (isWinner) classes.push('winner');
        if (isLoser) classes.push('loser');
        if (isAgentTeam(teamId)) classes.push('team-agent');
        return `
            <div class="${classes.join(' ')}">
                <div class="team-name">${teamLabel}</div>
                <div class="score">${scoreValue}</div>
                <div class="win-mark">${isWinner ? 'WIN' : (isTie ? 'TIE' : '')}</div>
            </div>
        `;
    }

    const homeRow = buildRow(homeLabel, homeValue, homeTeamId, homeWins, awayWins);
    const awayRow = buildRow(awayLabel, awayValue, awayTeamId, awayWins, homeWins);
    const rows = (awayWins || isTie) ? `${awayRow}${homeRow}` : `${homeRow}${awayRow}`;
    return `<div class="matchup-card">${rows}</div>`;
}

function initBottomTabs() {
    const tabs = document.querySelectorAll('.bottom-tab');
    const panels = document.querySelectorAll('.bottom-panel-content');
    if (!tabs.length) return;
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            tabs.forEach(item => item.classList.toggle('active', item.dataset.tab === target));
            panels.forEach(panel => {
                const isActive = panel.dataset.panel === target;
                panel.hidden = !isActive;
                panel.classList.toggle('active', isActive);
            });
        });
    });
}

function initSeasonTabs(container) {
    const tabs = container.querySelectorAll('.season-tab');
    const panels = container.querySelectorAll('.season-panel');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            tabs.forEach(item => item.classList.toggle('active', item.dataset.tab === target));
            panels.forEach(panel => panel.classList.toggle('active', panel.dataset.panel === target));
        });
    });
}

async function simulateSeason() {
    if (!seasonResultsDiv) return;
    seasonResultsDiv.innerHTML = '<div class="roster-card"><h4>Season Simulation</h4><p>Loading...</p></div>';
    try {
        const resp = await fetch('/api/simulate_season', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
        renderSeasonResults(data);
    } catch (e) {
        console.error('Season simulation error:', e);
        seasonResultsDiv.innerHTML = `<div class="roster-card"><h4>Season Simulation</h4><p style="color:red;">${e.message}</p></div>`;
    }
}

function renderSeasonResults(data) {
    const records = data.regular_season_records || [];
    const matchups = data.regular_season_matchups || [];
    const playoffResults = data.playoff_results || [];
    const winnerTeamId = data.winner_team_id;
    const winner = data.winner || formatTeamLabel(winnerTeamId, 'N/A');

    const standingsRows = records.map((rec, index) => {
        const rank = index + 1;
        const teamId = rec.team_id;
        const name = rec.team || formatTeamLabel(teamId, 'Unknown');
        const isChampion = winnerTeamId != null && teamId != null && Number(teamId) === Number(winnerTeamId);
        const championTag = isChampion ? '<span class="team-champion-tag">Champion</span>' : '';
        const rowClasses = [];
        if (isAgentTeam(teamId)) rowClasses.push('team-agent');
        if (rank === 1) rowClasses.push('standings-rank-1');
        else if (rank === 2) rowClasses.push('standings-rank-2');
        else if (rank === 3) rowClasses.push('standings-rank-3');
        if (isChampion) rowClasses.push('standings-champion');
        const rowClassAttr = rowClasses.length ? ` class="${rowClasses.join(' ')}"` : '';
        const W = rec.W ?? 0;
        const L = rec.L ?? 0;
        const T = rec.T ?? 0;
        const pts = rec.pts ?? 0;
        return `<tr${rowClassAttr}><td class="rank-cell">${formatStandingsRank(rank)}</td><td class="team-cell">${name}${championTag}</td><td class="num">${W}</td><td class="num">${L}</td><td class="num">${T}</td><td class="num">${Number(pts).toFixed(2)}</td></tr>`;
    }).join('');

    const matchupsByWeek = {};
    for (const matchup of matchups) {
        const week = matchup.week ?? 0;
        if (!matchupsByWeek[week]) matchupsByWeek[week] = [];
        matchupsByWeek[week].push(matchup);
    }
    const regularWeeks = Object.keys(matchupsByWeek).map(Number).sort((a, b) => a - b);
    const regularSeasonHTML = regularWeeks.length
        ? regularWeeks.map((week, index) => {
            const weekMatchups = matchupsByWeek[week]
                .sort((a, b) => (a.matchup || 0) - (b.matchup || 0))
                .map(game => renderMatchupCard(
                    game.away_team,
                    game.home_team,
                    game.away_score,
                    game.home_score,
                    game.away_team_id,
                    game.home_team_id
                ))
                .join('');
            const openAttr = index === 0 ? ' open' : '';
            return `
                <details class="week-accordion"${openAttr}>
                    <summary>Week ${week}</summary>
                    <div class="week-accordion-body">${weekMatchups}</div>
                </details>
            `;
        }).join('')
        : '<div style="opacity:0.7;">No regular-season matchups available.</div>';

    const resultsByWeek = {};
    for (const game of playoffResults) {
        const week = game.week ?? 0;
        if (!resultsByWeek[week]) resultsByWeek[week] = [];
        resultsByWeek[week].push(game);
    }
    const playoffWeeks = Object.keys(resultsByWeek).map(Number).sort((a, b) => a - b);
    const roundTitles = ['Quarterfinals', 'Semifinals', 'Final'];
    const playoffsHTML = playoffWeeks.length
        ? playoffWeeks.map((week, index) => {
            const title = roundTitles[index] || `Round ${index + 1}`;
            const gamesHTML = resultsByWeek[week]
                .sort((a, b) => (a.matchup || 0) - (b.matchup || 0))
                .map(game => renderMatchupCard(
                    game.away_team,
                    game.home_team,
                    game.away_score,
                    game.home_score,
                    game.away_team_id,
                    game.home_team_id
                ))
                .join('');
            return `
                <div class="playoff-card">
                    <h6>${title}</h6>
                    ${gamesHTML || '<div style="opacity:0.7;">No games</div>'}
                </div>
            `;
        }).join('')
        : '<div style="opacity:0.7;">No playoff games.</div>';

    const html = `
        <div class="roster-card">
            <h4>Season Simulation Results</h4>
            <div class="champion-banner">
                <span class="champion-label">Champion</span>
                <span class="champion-team">${winner}</span>
            </div>
            <div class="season-tabs">
                <button type="button" class="season-tab active" data-tab="standings">Standings</button>
                <button type="button" class="season-tab" data-tab="regular">Regular Season</button>
                <button type="button" class="season-tab" data-tab="playoffs">Playoffs</button>
            </div>
            <div class="season-panel active" data-panel="standings">
                <div style="overflow:auto;">
                    <table class="standings-table">
                        <thead>
                            <tr>
                                <th class="rank-cell">Rank</th>
                                <th>Team</th>
                                <th class="num">W</th>
                                <th class="num">L</th>
                                <th class="num">T</th>
                                <th class="num">Pts For</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${standingsRows}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="season-panel" data-panel="regular">
                ${regularSeasonHTML}
            </div>
            <div class="season-panel" data-panel="playoffs">
                <div class="playoff-rounds">${playoffsHTML}</div>
            </div>
        </div>
    `;
    seasonResultsDiv.innerHTML = html;
    initSeasonTabs(seasonResultsDiv);
}
