# Player Insights — Part 2 Plan: Live Draft Advisor & UI Pool Filtering

## Status

**Planned — not yet implemented.** Depends on Part 1 offline insight artifacts (`data/player_insights_{year}.json`) and the two manual Docker Compose enrichment services (search cache + Gemini synthesis).

Part 2 adds a real-time **LLM draft co-manager** and tightens **UI-side player pool filtering** so fragile veterans can be hidden without mutating the catalog, VORP baselines, or RL training data.

---

## Goals

1. **Live pick advice** — On demand, recommend the next best player for the user's team with short, grounded reasoning.
2. **UI pool filtering** — Let the user hide low-durability veterans via the existing **Min GP Frac** control without changing `load_player_catalog` or `DraftGymEnv`.
3. **Research override** — Surface high-value sleepers who fall below the GP threshold when offline insights confirm injury recovery.
4. **Preserve existing RL suggestions** — The header position-probability chips (`/api/draft/ai_suggestion_for_team`) stay fast and model-backed; the LLM advisor is a separate, slower, explainable layer.

---

## Non-Goals (v1)

- Multi-turn LLM tool use or live web search during a pick.
- Server-side pool mutation at catalog load time (no training-serving skew).
- Automatic insight refresh (Part 1 is entirely manual).
- Replacing the RL inference provider or bot simulation logic.

---

## Relationship to Part 1

| Part 1 delivers | Part 2 consumes |
| --- | --- |
| `data/player_insights_{year}.json` keyed by `sleeper_id` | `outlook_phrase`, `summary`, `tags`, `depth_role`, `playing_time_tier`, `injury_risk`, `recovery_status`, `bullets`, `fields_unknown` |
| `data/cache/insights/search/{sleeper_id}/` raw CSE payloads | Not read at runtime (audit / re-synthesis only) |
| Top **150** players by ADP enriched | Advisor context limited to enriched players; others show `"—"` in UI |

The webapp loads insights once at startup (or on first request) alongside the player catalog. Missing insight records are valid — the advisor and UI degrade gracefully.

---

## Architectural Principles

1. **Deterministic math in Python, synthesis in the LLM** — VORP, roster needs, bye conflicts, positional scarcity, and pool membership are computed before the prompt is built.
2. **Single-turn advisor** — One markdown context payload in, one structured JSON recommendation out. No agentic loops under a draft clock.
3. **Filter in the UI, not the catalog** — `PlayerCatalog` and `available_player_ids` remain complete. Filtering affects display, advisor candidate sets, and optionally RL suggestion ignore lists — not persisted state.
4. **Rookies always pass GP filter** — `games_played_frac === "R"` bypasses Min GP Frac (bug fix applied in frontend).
5. **Explicit unknowns** — When Part 1 marked a field in `fields_unknown`, the advisor must say so rather than infer.

---

## Current Codebase Touchpoints

### Already exists

| Component | Location | Role today |
| --- | --- | --- |
| Player table + GP filter | `frontend/index.html` (`#gp-frac-min`, `fetchPlayers`) | Client-side filter on `games_played_frac` |
| Per-player blind checkbox | `frontend/index.html` (`blindSet`) | Excludes player from RL `ai_suggestion_for_team` via `ignore` query param |
| RL position suggestions | `/api/draft/ai_suggestion_for_team` | Fast QB/RB/WR/TE probability chips in header |
| `InferenceProvider` ABC | `src/draft_buddy/core/inference_provider.py` | RL-backed via `RlInferenceProvider` in `scripts/run_webapp.py` |
| `get_ui_state()` | `src/draft_buddy/web/session.py` | Full draft board, rosters, bye weeks, pick clock |
| `/api/players` | `src/draft_buddy/web/app.py` | Available players + live VORP + Sleeper status fields |
| `Player` entity | `src/draft_buddy/core/entities.py` | `games_played_frac`, Sleeper injury/depth fields |

### Does not exist yet

- Insight JSON loader and join in API responses.
- Research-override filter logic.
- `/api/draft/advisor` endpoint and Gemini client.
- Advisor UI panel / button.
- `outlook_phrase` column and row expand for insight detail.

---

## UI Pool Filtering (Durability Gate)

### Min GP Frac control (existing, refined)

**Location:** `frontend/index.html` — `#gp-frac-min` input, applied in `fetchPlayers()` after `/api/players` returns.

**Rules (v1):**

```
function passesGpFilter(player, gpMin):
    if player.games_played_frac === "R":
        return true                          # rookies always visible

    if hasResearchOverride(player):
        return true                          # see below

  if gpMin is empty:
        return true

    gp = Number(player.games_played_frac)
    if not finite(gp):
        return false

    return gp >= gpMin
```

**Default:** Empty (no filter). User may set e.g. `0.70` before/during draft.

**Visual cues:**

- Players hidden by GP filter: removed from table (current behavior).
- Players shown only via research override: badge on row, e.g. `Research` chip next to name.
- Players below threshold without override: not shown (same as today).

### Research override (new)

A player below the GP threshold is **unblinded** when offline insights satisfy all of:

```python
insight.recovery_status == "recovered"
and "injury_recovery" in insight.tags
and insight.overall_confidence in ("high", "medium")
```

**Examples from 2026 data:** Christian McCaffrey (`games_played_frac ≈ 0.24`), Rashee Rice (`≈ 0.15`) — high ADP but low historical availability; research may justify keeping them visible.

**Implementation:** Frontend needs insights joined onto player payloads (see API changes). Override logic lives in `fetchPlayers()` filter function.

### Manual blind checkbox (existing, unchanged)

`blindSet` continues to exclude specific `player_id`s from RL header suggestions only. It does **not** remove players from the table or the LLM advisor unless we explicitly wire that later.

**Future option:** Checkbox label could clarify "Exclude from AI suggestions" vs a separate "Hide from pool" — out of scope for v1.

### What we are NOT doing

- No `ENABLE_DURABILITY_POOL_FILTER` at `load_player_catalog` time.
- No changes to `DraftGymEnv`, VORP baselines, or `FeatureExtractor` for v1.
- Optional server-side filter flag documented here for a later training-serving parity pass if desired.

---

## Offline Insights in the UI (Part 2 scope)

### Player table

| Column | Source | Notes |
| --- | --- | --- |
| Outlook | `insight.outlook_phrase` | Max 8 words; `"—"` if missing |
| Info (expand) | `summary`, `bullets`, `tags` | Click row or info icon |

### Tag chips (controlled vocabulary from Part 1)

Display as small chips: `injury_recovery`, `role_expansion`, `committee`, etc. Color `injury_risk: high` subtly (e.g. amber row accent).

### `/api/players` enrichment

Extend payload per player:

```json
{
  "player_id": 4034,
  "name": "Christian McCaffrey",
  "...": "...",
  "insight": {
    "outlook_phrase": "Full-go, bellcow if healthy",
    "summary": "...",
    "tags": ["injury_recovery"],
    "depth_role": "starter",
    "playing_time_tier": "high",
    "injury_risk": "medium",
    "recovery_status": "recovered",
    "overall_confidence": "medium",
    "fields_unknown": [],
    "bullets": [{ "text": "...", "source_url": "..." }]
  }
}
```

`insight: null` when no record exists for that `sleeper_id`.

### Insight loader (new module)

- `src/draft_buddy/data/player_insights.py` — `load_player_insights(year: int) -> dict[int, PlayerInsight]`
- Path: `data/player_insights_{year}.json` (or year from config / env `DRAFT_YEAR=2026`)
- Loaded once per `DraftSessionManager` / app factory; thread-safe read.

---

## Live LLM Draft Advisor

### Trigger

**On-demand** — User clicks **"Ask Advisor"** (or similar) in the header when it is their team's pick (or any time for analysis).

Not polled on every pick sync tick. Typical latency budget: 2–5 seconds (Gemini Flash).

### Endpoint

```
POST /api/draft/advisor
```

**Request body (optional):**

```json
{
  "team_id": 10,
  "gp_min": 0.70,
  "max_candidates": 12
}
```

Defaults: `team_id` = session agent team (or team on clock), `gp_min` from client state, `max_candidates` = 12.

**Response:** Structured JSON (Pydantic-validated):

```json
{
  "recommended_player_id": 9221,
  "recommended_name": "Jahmyr Gibbs",
  "confidence": "high",
  "rationale_bullets": [
    "Best VORP among available RBs with high playing_time_tier.",
    "Fills open RB starter slot; no week-6 bye conflict with your WR core."
  ],
  "alternates": [
    { "player_id": 9509, "name": "Bijan Robinson", "reason": "Higher ceiling, slightly lower VORP at this pick." }
  ],
  "flags": ["none"],
  "unknown_factors": []
}
```

**Errors:** `503` if insights file missing; `502` if Gemini fails; always return deterministic fallback message suggesting RL chips.

### Advisor service (new)

`src/draft_buddy/web/draft_advisor.py` (or `src/draft_buddy/advisor/` if it grows):

```text
DraftAdvisorService
├── build_context(session, team_id, gp_min, insights) -> str   # markdown
├── recommend(context) -> PickRecommendation                   # Gemini call
└── apply_pool_filter(players, gp_min, insights) -> list       # same rules as UI
```

**Dependency injection:** `GeminiAdvisorGateway` ABC + `GeminiFlashGateway` impl; API key via `GEMINI_API_KEY` env.

### Context payload (markdown, built in Python)

Sections the service assembles before the LLM sees anything:

```markdown
## Draft clock
- Pick 47 (round 4, pick 11)
- Team on clock: Michael Bertagna (team 10)
- Agent team: 10

## Your roster
| slot | player | pos | proj | bye |
...

## Positional needs (computed)
- Starters open: WR x1, FLEX x1
- Bench: RB depth optional

## Bye week pressure (weeks 5-14)
- Heavy: 11 (3 starters)
- Conflicts if drafting: [list players with bye 11 among top candidates]

## Positional baselines & scarcity
- QB baseline: 18.2 | available QB count above baseline: 4
- RB baseline: 12.1 | ...

## Available candidates (pool after GP filter + research override)
Sorted by VORP desc, max 12 rows.

| player_id | name | pos | vorp | adp | gp_frac | outlook | depth_role | playing_time | injury_risk | recovery | tags | confidence | unknown_fields |
...

## Instructions
- Recommend exactly one player from the table.
- Cite insight outlook/summary when relevant.
- If fields_unknown is non-empty for a candidate, mention "insufficient reporting on X".
- Do not recommend players not in the table.
- Output JSON matching PickRecommendation schema only.
```

**Deterministic pre-computation (not LLM):**

- VORP from `session.get_positional_baselines()`
- Roster slot needs from `categorize_roster_by_slots` / roster counts vs `ROSTER_STRUCTURE`
- Bye aggregation from `get_ui_state()["team_bye_weeks"]` cross-referenced with candidate bye weeks
- Pool filter identical to UI (`passesGpFilter` + research override)
- Candidate cap: top N by VORP at need positions first, then fill with best overall VORP

### Gemini contract

- Model: `gemini-2.0-flash` (configurable)
- Structured output via Pydantic `response_schema` / JSON mode
- System prompt: single-turn analyst; no tools; must pick from candidate table
- Temperature: low (0.2–0.3)

### Coexistence with RL suggestions

| Feature | Speed | Output | Use case |
| --- | --- | --- | --- |
| RL header chips | ~100ms | QB/RB/WR/TE % | Glance at model lean |
| LLM advisor | ~2–5s | Named player + why | Decision support at critical picks |
| Manual blind | instant | Excludes from RL ignore set | Personal landmine list |

The advisor candidate pool respects GP filter + research override but **not** `blindSet` unless we add an optional `respect_blind_set: true` flag later.

---

## Frontend Changes

### Header

- Add **"Ask Advisor"** button near `#ai-suggestion-display`.
- On click: `POST /api/draft/advisor` with current `gpMin` and agent team.
- Show result in a dismissible panel: recommended name, 2–3 rationale bullets, alternates, flags.
- Loading state + error toast.

### Player table

- New column **Outlook** (`outlook_phrase`).
- Research override badge on name cell.
- Row expand or info icon → `summary`, linked `bullets`, tag chips.

### GP filter fix (done)

Rookies (`games_played_frac === "R"`) now return `true` from the GP min filter instead of being excluded.

---

## Configuration & Environment

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Advisor + Part 1 synthesis |
| `DRAFT_YEAR` | Default `2026` for insight file path |
| `PLAYER_INSIGHTS_PATH` | Optional override for `data/player_insights_{year}.json` |
| `ADVISOR_MODEL` | Default `gemini-2.0-flash` |
| `ADVISOR_MAX_CANDIDATES` | Default `12` |

Add to `docker-compose.yml` `webapp` service `environment` block when implementing.

---

## File Layout (new / modified)

```text
src/draft_buddy/
├── data/
│   └── player_insights.py          # load + lookup by sleeper_id
├── advisor/                        # optional package
│   ├── __init__.py
│   ├── schemas.py                  # PickRecommendation, AdvisorContext
│   ├── context_builder.py          # markdown assembly
│   └── gemini_gateway.py           # Gemini Flash structured call
└── web/
    ├── app.py                      # + insight join on /api/players, POST /api/draft/advisor
    └── session.py                  # optional: expose agent_team_id helper

frontend/
└── index.html                      # outlook column, advisor UI, GP + override filter

data/
└── player_insights_2026.json       # produced by Part 1 (manual)

tests/
├── test_player_insights_loader.py
├── test_advisor_context_builder.py
├── test_advisor_pool_filter.py
└── test_web_advisor_endpoint.py    # mocked Gemini
```

---

## Sequencing (Part 2 implementation order)

1. **`player_insights.py` loader** + tests with fixture JSON (no Gemini).
2. **Join insights on `/api/players`** — frontend outlook column (read-only).
3. **Research override in `fetchPlayers`** — requires step 2.
4. **`DraftAdvisorService` context builder** — pure Python tests against fixture session state.
5. **`GeminiAdvisorGateway`** + `POST /api/draft/advisor` — mocked in tests.
6. **Frontend advisor panel** + button wiring.
7. **Docs** — README section for advisor usage and env vars.

Part 2 can start as soon as Part 1 has produced at least a partial `player_insights_2026.json` for testing (even a 5-player fixture).

---

## Testing Strategy

| Layer | Approach |
| --- | --- |
| Pool filter | Unit tests: rookie pass-through, GP threshold, override conditions, null insight |
| Context builder | Snapshot or assert key markdown sections contain expected VORP/roster rows |
| Gemini gateway | Mock HTTP; assert schema validation on response |
| Web endpoint | `TestClient` with injected mock advisor service |
| Frontend | Manual: set GP 0.70, verify CMC appears with override badge when insight qualifies |

---

## Edge Cases

| Case | Behavior |
| --- | --- |
| No insights file | `/api/players` returns `insight: null`; advisor returns 503 with message to run Part 1 enrichment |
| Player not in top 150 enriched set | No insight; still draftable; advisor uses stats-only row |
| Gemini timeout | 502 + suggest using RL chips |
| Synced Sleeper session (future) | Advisor read-only; uses same `get_ui_state` / available pool |
| Empty candidate pool after filter | Advisor returns explicit "no players pass your filters" without calling Gemini |
| Conflicting snippets (Part 1) | `overall_confidence: low` — advisor should mention uncertainty in `unknown_factors` |

---

## Future Extensions (explicitly deferred)

- **Server-side pool filter** behind `Config.draft.ENABLE_DURABILITY_POOL_FILTER` for RL training parity.
- **Auto-advisor** on pick clock (user preference).
- **Sleeper live sync** (`SLEEPER_LIVE_DRAFT_SYNC_PLAN.md`) — advisor works unchanged on read-only synced sessions.
- **Streaming advisor response** for faster perceived latency.
- **User-editable landmine list** persisted to localStorage (beyond `blindSet` session scope).

---

## Open Questions (resolve during implementation)

1. **Advisor team default** — Always agent team (10) or team on clock? Recommend: agent team with optional override in request body.
2. **Insight year** — Hardcode 2026 vs derive from config / player data metadata?
3. **Candidate selection** — Pure top VORP vs positional-need-weighted shortlist? Recommend: 60% need positions, 40% best VORP fill for diversity.
4. **Advisor during opponent picks** — Allow for trade bait analysis or disable button? Recommend: allow with clear "not your pick" banner.

---

## Summary

Part 2 layers a **single-turn Gemini advisor** and **richer UI filtering** on top of Part 1's offline research artifacts without touching catalog load or RL training. Python owns the math and pool rules; the LLM explains the pick. The existing RL probability chips remain the fast baseline; the advisor is the explainable co-manager invoked when the user wants a reasoned recommendation.
