// script.js

const API_BASE_URL = "http://127.0.0.1:8000";

// --- DOM Elements ---
const playerListEl = document.getElementById('player-list');
const currentPickInfoEl = document.getElementById('current-pick-info');
const agentSuggestionEl = document.getElementById('agent-suggestion');
const teamButtonsEl = document.getElementById('team-buttons');
const rostersViewEl = document.getElementById('rosters-view');
const summaryViewEl = document.getElementById('summary-view');
const undoBtn = document.getElementById('undo-btn');
const exportBtn = document.getElementById('export-btn');
const searchBar = document.getElementById('search-bar');
const positionFiltersEl = document.getElementById('position-filters');


// --- State ---
let draftState = {};
let selectedTeamId = null;

// --- API Functions ---
async function fetchState() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/state`);
        draftState = await response.json();
        updateUI();
        fetchSuggestion();
    } catch (error) {
        console.error("Failed to fetch state:", error);
        currentPickInfoEl.textContent = "Error connecting to server.";
    }
}

async function fetchSuggestion() {
     try {
        const response = await fetch(`${API_BASE_URL}/api/suggestion`);
        const data = await response.json();
        agentSuggestionEl.textContent = `Agent Suggests: ${data.suggestion}`;
    } catch (error) {
        console.error("Failed to fetch suggestion:", error);
    }
}

async function postPick(playerId, teamId) {
    await fetch(`${API_BASE_URL}/api/pick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_id: playerId, team_id: teamId })
    });
    fetchState(); // Refresh state after action
}

async function undoLastPick() {
    await fetch(`${API_BASE_URL}/api/undo-last-pick`, { method: 'POST' });
    fetchState(); // Refresh state
}


// --- Render Functions ---
function updateUI() {
    renderPlayerList();
    renderControlPanel();
    renderTeamRosters();
    renderDraftSummary();
}

function renderPlayerList() {
    playerListEl.innerHTML = ''; // Clear existing list
    
    const searchTerm = searchBar.value.toLowerCase();
    const checkedPositions = [...positionFiltersEl.querySelectorAll('input:checked')].map(cb => cb.value);

    const filteredPlayers = draftState.available_players.filter(p => {
        const nameMatch = p.name.toLowerCase().includes(searchTerm);
        const positionMatch = checkedPositions.length === 0 || checkedPositions.includes(p.position);
        return nameMatch && positionMatch;
    });

    filteredPlayers.forEach(player => {
        const item = document.createElement('div');
        item.className = 'player-item';
        item.dataset.playerId = player.player_id;
        item.innerHTML = `
            <span class="position">${player.position}</span>
            <span class="name">${player.name}</span>
            <span class="points">${player.projected_points.toFixed(1)} pts</span>
        `;
        playerListEl.appendChild(item);
    });
}

function renderControlPanel() {
    if (!draftState.draft_order || draftState.draft_order.length === 0) {
        currentPickInfoEl.textContent = "Draft not initialized.";
        return;
    }

    const currentPick = draftState.current_pick_overall;
    if (currentPick > draftState.draft_order.length) {
        currentPickInfoEl.textContent = "Draft Complete!";
        agentSuggestionEl.textContent = "";
        return;
    }

    const teamOnTheClock = draftState.draft_order[currentPick - 1];
    const round = Math.floor((currentPick - 1) / draftState.config.num_teams) + 1;
    const pickInRound = ((currentPick - 1) % draftState.config.num_teams) + 1;
    currentPickInfoEl.textContent = `Round ${round}, Pick ${pickInRound} (Overall: ${currentPick}): Team ${teamOnTheClock} is on the clock.`;
    
    // Set the default selected team if none is manually chosen
    if (selectedTeamId === null) {
        selectedTeamId = teamOnTheClock;
    }
    
    renderTeamButtons(teamOnTheClock);
}

function renderTeamButtons(teamOnTheClock) {
    teamButtonsEl.innerHTML = '';
    for (let i = 1; i <= draftState.config.num_teams; i++) {
        const btn = document.createElement('button');
        btn.className = 'team-btn';
        btn.dataset.teamId = i;
        btn.textContent = `Team ${i}`;
        
        // Highlight based on manual selection OR the team on the clock
        if (selectedTeamId === i) {
             btn.classList.add('highlight');
        } else if (selectedTeamId === null && teamOnTheClock === i){
            btn.classList.add('highlight');
        }
        
        teamButtonsEl.appendChild(btn);
    }
}

function renderTeamRosters() {
    rostersViewEl.innerHTML = '';
    if (!draftState.teams) return;

    Object.values(draftState.teams).forEach(team => {
        const card = document.createElement('div');
        card.className = 'roster-card';
        let rosterHtml = `<h5>Team ${team.team_id}</h5><ul>`;
        team.roster.forEach(p => {
            rosterHtml += `<li><strong>${p.position}</strong> - ${p.name}</li>`;
        });
        rosterHtml += '</ul>';
        card.innerHTML = rosterHtml;
        rostersViewEl.appendChild(card);
    });
}

function renderDraftSummary() {
    const counts = { QB: 0, RB: 0, WR: 0, TE: 0 };
    draftState.picks.forEach(pick => {
        const pos = pick.player.position;
        if (pos in counts) {
            counts[pos]++;
        }
    });
    summaryViewEl.innerHTML = `
        <ul>
            <li>QBs Drafted: ${counts.QB}</li>
            <li>RBs Drafted: ${counts.RB}</li>
            <li>WRs Drafted: ${counts.WR}</li>
            <li>TEs Drafted: ${counts.TE}</li>
        </ul>
    `;
}

// --- Event Listeners ---
function initialize() {
    document.addEventListener('DOMContentLoaded', fetchState);

    undoBtn.addEventListener('click', undoLastPick);
    
    exportBtn.addEventListener('click', () => {
        window.open(`${API_BASE_URL}/api/export`);
    });

    searchBar.addEventListener('input', renderPlayerList);
    positionFiltersEl.addEventListener('change', renderPlayerList);

    playerListEl.addEventListener('click', (e) => {
        const playerItem = e.target.closest('.player-item');
        if (playerItem) {
            const playerId = playerItem.dataset.playerId;
            if (selectedTeamId !== null) {
                postPick(playerId, selectedTeamId);
                selectedTeamId = null; // Reset selection after pick
            } else {
                alert("Please select a team first.");
            }
        }
    });
    
    teamButtonsEl.addEventListener('click', (e) => {
        const teamBtn = e.target.closest('.team-btn');
        if(teamBtn) {
            selectedTeamId = parseInt(teamBtn.dataset.teamId, 10);
            renderTeamButtons(); // Re-render to show new highlight
        }
    });
}

initialize();