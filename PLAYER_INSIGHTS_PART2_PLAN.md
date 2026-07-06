# Player Insights — Part 2 Plan: Live Draft Assistant

## Status

**Assistant implemented.** Part 1 offline enrichment and UI insight display are done. The live LLM draft assistant runs via `POST /api/draft/advisor` when `GEMINI_API_KEY` is set.

**Already shipped (no Part 2 work):**

| Component | Location |
| --- | --- |
| Insight loader + latest export resolution | `src/draft_buddy/data/insights/loader.py`, `cache_paths.py` |
| Insights loaded at webapp startup | `scripts/run_webapp.py` |
| `/api/players` insight join | `src/draft_buddy/web/app.py` |
| `/api/insights/meta` | `src/draft_buddy/web/app.py` |
| Outlook / Role columns, insight modal | `frontend/index.html` |
| Min GP Frac filter (client-side, rookie bypass) | `frontend/index.html` |
| Blind checkbox → RL ignore list | `frontend/index.html` |

**Out of scope for this revision:** automatic research-override GP filtering, research badges, server-side catalog mutation, RL/training changes.

---

## Goals

1. **Live pick advice** — On demand, recommend the best next player for the **advising team** with short, grounded reasoning.
2. **Full manual filtering control** — User controls what they see (Min GP Frac) and what the RL model sees (blind checkboxes). No automatic un-hiding or pool overrides.
3. **Token-conscious usage** — Master UI toggle enables/disables the assistant so Gemini is never called unless the user opts in.
4. **Preserve RL suggestions** — Header position chips (`/api/draft/ai_suggestion_for_team`) stay fast and model-backed; the assistant is a separate, slower, explainable layer.

---

## Non-Goals (v1)

- Multi-turn LLM tool use or live web search during a pick.
- Automatic research-override GP filtering or `Research` badges.
- Server-side pool mutation at catalog load time.
- Automatic insight refresh (Part 1 remains manual).
- Replacing the RL inference provider or bot simulation logic.
- Advisor respecting `blindSet` (explicitly ignored for v1).

---

## Decisions (resolved)

| Question | Decision |
| --- | --- |
| Insights file required? | **No.** Advisor works stats-only when insights are missing; enriched fields are included when available. |
| Advising team | **Team on clock** (`current_team_picking`), which reflects UI board-header overrides via `/api/draft/override_team`. Optional `team_id` in request body for explicit override. |
| Agent / “my team” | From **`config.draft.AGENT_START_POSITION`** (e.g. team **2** for `red_league_10` 2026). Exposed in UI state as `agent_start_position`. |
| Pick scope | UI toggle: **My picks only** vs **Every team**. When “My picks only”, advisor is available only when the advising team equals `agent_start_position`. |
| Candidate shortlist | **Top K by VORP and top K by ADP per position** (QB/RB/WR/TE). Include a field glossary in the prompt so the LLM understands each column. |
| Insight year / file path | Derive from **`config.season.season`** via `load_runtime_config()`. Loader already resolves newest export under `data/insights/exports/` with legacy fallback. No separate `DRAFT_YEAR` env needed. |
| Gemini model | **`ADVISOR_GEMINI_MODEL`** env (default `gemini-2.5-flash`), separate from **`INSIGHTS_GEMINI_MODEL`** used by Part 1 synthesis. |
| Pool filter for advisor | **Shared Python module** under `web/`; client sends `gp_min` from the Min GP Frac input. Same rules as UI: rookie pass-through, numeric threshold, **no research override**. |
| Package layout | All new advisor code under **`src/draft_buddy/web/`**. |
| Test data | ~10 players in `data/player_insights_2026.json` is sufficient for v1 dev. |

---

## Architectural Principles

1. **Deterministic math in Python, synthesis in the LLM** — VORP, roster needs, bye conflicts, positional baselines, and pool membership are computed before the prompt is built.
2. **Single-turn assistant** — One markdown context payload in, one structured JSON recommendation out. No agentic loops under a draft clock.
3. **Filter in the UI, not the catalog** — `PlayerCatalog` and `available_player_ids` remain complete. GP filter and blind list are user-controlled client concerns; advisor receives `gp_min` as a hint, not a server-side catalog change.
4. **Explicit unknowns** — When Part 1 marked a field in `fields_unknown`, the assistant must say so rather than infer.
5. **No surprise token spend** — Assistant master toggle off → no Gemini calls, button disabled/hidden.

---

## Manual Filtering (unchanged behavior)

### Min GP Frac (existing)

Client-side filter in `fetchPlayers()` after `/api/players` returns.

```
passesGpFilter(player, gpMin):
    if player.games_played_frac == "R": return true   # rookies always visible
    if gpMin is empty: return true
    gp = Number(player.games_played_frac)
    if not finite(gp): return false
    return gp >= gpMin
```

User sets e.g. `0.70` to hide fragile veterans. **No automatic research override** — if the user wants CMC visible despite low GP, they lower or clear the threshold manually.

### Blind checkbox (existing)

`blindSet` excludes `player_id`s from RL header suggestions via the `ignore` query param on `/api/draft/ai_suggestion_for_team`. Does **not** affect the assistant candidate pool in v1.

---

## Live LLM Draft Assistant

### Master toggle (new)

**Location:** Header controls near the AI chip.

| State | Behavior |
| --- | --- |
| **Off** (default) | “Ask Assistant” hidden/disabled. No `/api/draft/advisor` calls. |
| **On** | “Ask Assistant” available per scope rules below. |

Persist toggle in `sessionStorage` so it survives page refresh during a draft session.

### Pick scope toggle (new)

**Location:** Adjacent to master toggle.

| Mode | “Ask Assistant” enabled when |
| --- | --- |
| **My picks only** | Advising team (`current_team_picking` or request `team_id`) == `agent_start_position` |
| **Every team** | Any advising team (useful for trade-bait / opponent analysis) |

When scope blocks the action, show a short banner (e.g. “Assistant available on your picks only — switch scope or wait for your turn”) rather than calling Gemini.

### Trigger

**On-demand button click** — User clicks **“Ask Assistant”** when the master toggle is on and scope allows.

Not polled on every pick sync. Typical latency budget: 2–5 seconds.

### Endpoint

```
POST /api/draft/advisor
```

**Request body:**

```json
{
  "team_id": 2,
  "gp_min": 0.70,
  "top_k": 5
}
```

| Field | Default | Notes |
| --- | --- | --- |
| `team_id` | `current_team_picking` | Reflects UI clock override when user clicks a board header |
| `gp_min` | omitted / null | No GP filter applied server-side |
| `top_k` | `5` (or `ADVISOR_TOP_K_PER_POSITION`) | Per-position cap for each ranking (VORP and ADP) |

**Response:** Structured JSON (Pydantic-validated):

```json
{
  "advising_team_id": 2,
  "is_agent_team": true,
  "recommended_player_id": 9221,
  "recommended_name": "Jahmyr Gibbs",
  "confidence": "high",
  "rationale_bullets": [
    "Top RB by VORP in the shortlist with high playing_time_tier.",
    "Fills open RB starter slot; no week-6 bye conflict with your WR core."
  ],
  "alternates": [
    { "player_id": 9509, "name": "Bijan Robinson", "reason": "Higher ADP, slightly lower VORP." }
  ],
  "flags": ["none"],
  "unknown_factors": []
}
```

**Errors:**

| Code | When |
| --- | --- |
| `400` | Invalid team id or empty candidate pool after filters |
| `403` | Scope is “My picks only” and advising team ≠ agent team (optional — may instead disable button client-side) |
| `502` | Gemini failure; message suggests using RL chips |
| `503` | Not used for missing insights (advisor degrades to stats-only) |

When candidate pool is empty after GP filter, return `400` with a clear message — do not call Gemini.

### Service layout (under `web/`)

```text
src/draft_buddy/web/
├── draft_advisor_schemas.py    # PickRecommendation, AdvisorRequest
├── draft_advisor_filter.py     # shared GP filter (Python mirror of frontend)
├── draft_advisor_context.py    # markdown context assembly + field glossary
├── draft_advisor_gateway.py    # GeminiAdvisorGateway ABC + Flash impl
├── draft_advisor_service.py    # orchestrates filter → context → recommend
└── app.py                      # POST /api/draft/advisor
```

```text
DraftAdvisorService
├── apply_gp_filter(players, gp_min) -> list
├── build_candidate_shortlists(players, baselines, top_k) -> dict[str, list]
├── build_context(session, team_id, candidates, insights) -> str
└── recommend(context) -> PickRecommendation
```

**Dependency injection:** `GeminiAdvisorGateway` ABC; API key via `GEMINI_API_KEY`; model via `ADVISOR_GEMINI_MODEL`.

### Context payload (markdown, built in Python)

Sections assembled before the LLM sees anything:

```markdown
## Field glossary
- **vorp**: Value Over Replacement Player — projected_points minus positional baseline.
- **adp**: Average draft position; lower = drafted earlier.
- **gp_frac**: Fraction of games played last season; "R" = rookie (no NFL sample).
- **outlook_phrase**: Short research summary (offline, may be missing).
- **depth_role**: starter | co_starter | committee | backup | unknown
- **playing_time_tier**: high | medium | low | unknown
- **injury_risk / recovery_status**: From offline research when available.
- **fields_unknown**: Insight fields with insufficient reporting — do not infer these.

## Draft clock
- Pick 47 (round 4, pick 11)
- Advising team: Goofy's Kitchen (team 2)
- Agent team: 2 | On clock: yes

## Advising team roster
| slot | player | pos | proj | bye |
...

## Positional needs (computed)
- Starters open: WR x1, FLEX x1
- Bench: RB depth optional

## Bye week pressure (weeks 4–14)
- Week 11: 3 starters on bye (RB:1, WR:2)
- Candidate bye conflicts: [players whose bye_week matches heavy weeks]

## Positional baselines
| pos | baseline | available above baseline |
...

## Top candidates by position
Each table: top K by VORP and top K by ADP (deduplicated within position).

### RB — by VORP
| player_id | name | vorp | adp | gp_frac | proj | bye | outlook | depth_role | playing_time | injury_risk | recovery | tags | confidence | fields_unknown |
...

### RB — by ADP
...

(repeat for QB, WR, TE)

## Instructions
- Recommend exactly one player from the candidate tables above.
- Prefer need-filling picks when VORP is close.
- Cite insight outlook/summary when present; use stats only when insight is null.
- If fields_unknown is non-empty, mention insufficient reporting — do not guess.
- Do not recommend players not listed in the candidate tables.
- Output JSON matching PickRecommendation schema only.
```

**Deterministic pre-computation (not LLM):**

- VORP from `session.get_positional_baselines()`
- Roster slot needs from roster counts vs `ROSTER_STRUCTURE`
- Bye aggregation from `get_ui_state()["team_bye_weeks"]` for the **advising team**
- GP filter from request `gp_min` (shared Python module)
- Per-position shortlists: top K by VORP desc, top K by ADP asc (finite ADP only), merge/dedupe per position for the prompt tables

### Gemini contract

- Model: `ADVISOR_GEMINI_MODEL` (default `gemini-2.5-flash`)
- Structured output via Pydantic `response_schema` / JSON mode
- System prompt: single-turn draft analyst; no tools; must pick from candidate tables
- Temperature: low (0.2)

### Coexistence with RL suggestions

| Feature | Speed | Output | User control |
| --- | --- | --- | --- |
| RL header chips | ~100ms | QB/RB/WR/TE % | Blind checkboxes |
| LLM assistant | ~2–5s | Named player + why | Master toggle + scope toggle + manual click |
| Min GP Frac | instant | Hides table rows | User input |
| Blind checkbox | instant | RL ignore list | Per-player checkbox |

---

## Frontend Changes (assistant only)

### Header

- **Assistant enabled** master toggle (default off).
- **Scope** toggle: “My picks only” | “Every team”.
- **Ask Assistant** button — visible when master toggle on; enabled when scope allows.
- On click: `POST /api/draft/advisor` with `team_id` = `current_team_picking`, `gp_min` from `#gp-frac-min`, `top_k` from config/default.
- Dismissible result panel: recommended name, rationale bullets, alternates, flags.
- Loading spinner + error toast (502 → suggest RL chips).

When advising team ≠ agent team and scope is “My picks only”, show informational banner; do not enable the button.

---

## Configuration & Environment

| Variable | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Advisor + Part 1 synthesis | (required for live calls) |
| `ADVISOR_GEMINI_MODEL` | Assistant model | `gemini-2.5-flash` |
| `INSIGHTS_GEMINI_MODEL` | Part 1 synthesis only | `gemini-2.5-flash` |
| `ADVISOR_TOP_K_PER_POSITION` | Candidates per ranking per position | `5` |

**Not needed:** `DRAFT_YEAR` — season comes from `DRAFT_BUDDY_SEASON` / league season overlay (`config.season.season`). Insight file resolution uses `load_latest_player_insights(config.paths.DATA_DIR)`.

Optional future: `PLAYER_INSIGHTS_PATH` override for testing.

Add advisor env vars to `docker-compose.yml` `webapp` service when implementing.

---

## File Layout (new / modified)

```text
src/draft_buddy/web/
├── draft_advisor_schemas.py
├── draft_advisor_filter.py
├── draft_advisor_context.py
├── draft_advisor_gateway.py
├── draft_advisor_service.py
└── app.py                          # POST /api/draft/advisor

frontend/
└── index.html                      # assistant toggles, button, result panel

tests/
├── test_draft_advisor_filter.py
├── test_draft_advisor_context.py
└── test_web_advisor_endpoint.py    # mocked Gemini
```

---

## Sequencing

1. **`draft_advisor_filter.py`** — GP filter mirroring frontend; unit tests.
2. **`draft_advisor_context.py`** — per-position top-K shortlists, markdown assembly, field glossary; fixture session tests.
3. **`draft_advisor_schemas.py` + `draft_advisor_gateway.py`** — Pydantic models + mocked Gemini tests.
4. **`draft_advisor_service.py` + `POST /api/draft/advisor`** — wire session, insights, config; endpoint tests.
5. **Frontend** — master toggle, scope toggle, Ask Assistant button, result panel.
6. **Docs** — README section for assistant usage and env vars.

---

## Testing Strategy

| Layer | Approach |
| --- | --- |
| GP filter | Unit tests: rookie pass-through, threshold, empty gp_min, null gp_frac |
| Context builder | Assert markdown contains roster, bye weeks, per-position tables, glossary |
| Gemini gateway | Mock client; assert schema validation |
| Web endpoint | `TestClient` with injected mock service; scope + empty pool cases |
| Frontend | Manual: toggle off → no calls; my-picks-only blocks on opponent clock; every-team allows |

---

## Edge Cases

| Case | Behavior |
| --- | --- |
| No insights file | Stats-only candidate rows; `outlook` columns empty in prompt |
| Player not in enriched set | Still in candidate tables on VORP/ADP merit |
| Gemini timeout / error | 502 + suggest RL chips |
| UI team override (board header click) | `current_team_picking` updates; advisor uses new team |
| Empty pool after GP filter | 400, no Gemini call |
| Master toggle off | Button disabled; client never POSTs |
| Low insight confidence | Assistant mentions uncertainty in `unknown_factors` |

---

## Future Extensions (deferred)

- Research-override GP filtering and `Research` badges.
- Auto-assistant on pick clock (when master toggle on).
- Server-side pool filter for RL training parity.
- Advisor respecting `blindSet`.
- Streaming assistant response.
- Sleeper live sync (`SLEEPER_LIVE_DRAFT_SYNC_PLAN.md`).

---

## Open Questions (remaining)

All resolved for v1 implementation.

## Summary

Part 2 (revised) adds a **token-conscious, on-demand Gemini draft assistant** under `src/draft_buddy/web/`. Python owns VORP, roster needs, bye analysis, and GP filtering; the LLM picks from **per-position top-K VORP and ADP shortlists** with a field glossary. The user keeps full manual control of pool visibility and RL blind lists. RL position chips remain the fast baseline; the assistant is invoked only when the user enables it and clicks **Ask Assistant**.
