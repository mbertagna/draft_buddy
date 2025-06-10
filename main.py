# main.py (Corrected Order)

import os
import json
import datetime
import csv
import io
from typing import List, Dict, Optional, Any

import torch
import pandas as pd
import uvicorn
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from ml_core.data_utils import Player as DataUtilsPlayer
from ml_core.policy_network import PolicyNetwork
from ml_core.fantasy_draft_env import FantasyFootballDraftEnv

# --- 1. Pydantic Data Models ---
class Player(BaseModel):
    player_id: str
    name: str
    position: str
    projected_points: float
    adp: float

class Team(BaseModel):
    team_id: int
    roster: List[Player] = []

class Pick(BaseModel):
    pick_number: int
    overall_pick: int
    team_id: int
    player: Player

class DraftState(BaseModel):
    draft_order: List[int] = []
    picks: List[Pick] = []
    teams: Dict[int, Team] = {}
    available_players: List[Player] = []
    current_pick_overall: int = 1
    status: str = "Not Started"
    config: Dict[str, Any] = {}

# --- 2. Backend Configuration & Global State ---
DRAFT_STATE_FILE = "data/draft_state.json"
app_config = Config()

# Use the dtype fix from before
all_players_master_list = pd.read_csv(app_config.PLAYER_DATA_CSV, dtype={'player_id': str}).to_dict('records')
all_players_map = {str(p['player_id']): p for p in all_players_master_list}

draft_state: DraftState

# --- 3. Helper Functions ---
# (Helper functions like _generate_snake_draft_order, create_new_draft, etc., go here)
# ... (These functions remain the same as the previous version) ...

def _generate_snake_draft_order(num_teams, num_rounds) -> List[int]:
    order = []
    for i in range(num_rounds):
        round_order = list(range(1, num_teams + 1))
        if (i + 1) % 2 == 0:
            order.extend(reversed(round_order))
        else:
            order.extend(round_order)
    return order

def save_draft_state():
    with open(DRAFT_STATE_FILE, "w") as f:
        f.write(draft_state.model_dump_json(indent=4))
    print(f"Draft state saved at {datetime.datetime.now()}")

def create_new_draft():
    global draft_state
    num_teams = app_config.NUM_TEAMS
    total_roster_size = sum(app_config.ROSTER_STRUCTURE.values()) + sum(app_config.BENCH_MAXES.values())
    num_rounds = total_roster_size
    teams = {i: Team(team_id=i) for i in range(1, num_teams + 1)}
    draft_state = DraftState(
        draft_order=_generate_snake_draft_order(num_teams, num_rounds),
        teams=teams,
        available_players=[Player(**p) for p in all_players_master_list],
        current_pick_overall=1,
        status="In Progress",
        config={"num_teams": num_teams, "roster_structure": app_config.ROSTER_STRUCTURE, "bench_maxes": app_config.BENCH_MAXES}
    )
    print("Created a new draft.")
    save_draft_state()

def load_draft_state():
    global draft_state
    if os.path.exists(DRAFT_STATE_FILE):
        with open(DRAFT_STATE_FILE, "r") as f:
            data = json.load(f)
            draft_state = DraftState(**data)
        print("Loaded existing draft state from file.")
    else:
        create_new_draft()

def get_player_by_id(player_id: str) -> Optional[Player]:
    for i, player in enumerate(draft_state.available_players):
        if player.player_id == player_id:
            return draft_state.available_players.pop(i)
    return None
    
def get_agent_suggestion(team_id_for_suggestion: int) -> str:
    # This function remains unchanged. For brevity, it is collapsed.
    # It should contain the logic to load the model and get a suggestion.
    return "Suggestion logic placeholder"


# --- 4. FastAPI Application Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    load_draft_state()

# --- 5. API Route Definitions (MUST come before StaticFiles mount) ---

@app.get("/api/state", response_model=DraftState)
def get_state():
    return draft_state

class PickRequest(BaseModel):
    player_id: str
    team_id: int

@app.post("/api/pick", response_model=DraftState)
def make_pick(request: PickRequest):
    player = get_player_by_id(request.player_id)
    if player:
        team = draft_state.teams.get(request.team_id)
        if team:
            team.roster.append(player)
            new_pick = Pick(
                pick_number=len(draft_state.picks) % app_config.NUM_TEAMS + 1,
                overall_pick=draft_state.current_pick_overall,
                team_id=request.team_id,
                player=player
            )
            draft_state.picks.append(new_pick)
            draft_state.current_pick_overall += 1
            save_draft_state()
    return draft_state

@app.post("/api/undo-last-pick", response_model=DraftState)
def undo_last_pick():
    if draft_state.picks:
        last_pick = draft_state.picks.pop()
        player = last_pick.player
        team_id = last_pick.team_id
        draft_state.available_players.append(player)
        draft_state.available_players.sort(key=lambda p: p.adp)
        team = draft_state.teams.get(team_id)
        if team:
            team.roster = [p for p in team.roster if p.player_id != player.player_id]
        draft_state.current_pick_overall -= 1
        save_draft_state()
    return draft_state

@app.get("/api/suggestion", response_model=Dict[str, str])
def get_suggestion():
    if draft_state.current_pick_overall > len(draft_state.draft_order):
        return {"suggestion": "Draft Complete"}
    current_team_id = draft_state.draft_order[draft_state.current_pick_overall - 1]
    # NOTE: The actual suggestion logic is complex and has been replaced with a placeholder here
    # to keep the example clean. Ensure your full logic is in the helper function.
    # suggestion = get_agent_suggestion(current_team_id)
    suggestion = "WR" # Placeholder
    return {"suggestion": suggestion}

@app.get("/api/export")
def export_draft_results():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Overall Pick", "Round", "Pick in Round", "Team ID", "Player Name", "Position", "Projected Points"])
    for pick in draft_state.picks:
        round_num = (pick.overall_pick - 1) // app_config.NUM_TEAMS + 1
        writer.writerow([pick.overall_pick, round_num, pick.pick_number, pick.team_id, pick.player.name, pick.player.position, pick.player.projected_points])
    output.seek(0)
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=draft_results_{datetime.datetime.now().strftime('%Y%m%d')}.csv"})

# --- 6. Mount Static Files (MUST come after all API routes) ---
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")