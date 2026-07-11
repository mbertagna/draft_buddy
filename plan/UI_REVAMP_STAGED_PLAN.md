# Draft Buddy UI Revamp — Staged Delivery Plan

## Purpose

Deliver the UI revamp as small, behavior-preserving, individually committable stages so the
styling the user likes is adopted while functionality stays intact and verifiable at every
step. The entire `docs/DRAFT_BUDDY_UI_REVAMP_PLAN.md` (all three phases, including backend
changes) is in scope, along with the `frontend-design` principles.

## Two artifacts discovered (important context)

There were **two separate, uncommitted UI efforts**. Both are now preserved (nothing lost;
the stash and working tree are untouched):

1. **Phase 1 "sliver" work** — the working-tree monolithic `frontend/index.html`
   (HEAD + a compact "sliver" AI banner + settings drawer, with **working behavior**,
   including the override). Preserved to `plan/artifacts/phase1-sliver-index.html`.
2. **The real reskin** (the styling the user likes) — lived in `stash@{0}`, saved *with*
   its untracked files. It is a genuine, well-architected overhaul. Preserved to
   `plan/artifacts/reskin/`:
   - `styles/tokens.css` — a real **design-token system**: surfaces, brand greens, a
     spacing scale, radii, shadows, and deliberate typography (DM Sans / IBM Plex Sans /
     IBM Plex Mono), with the rainbow lava gradient scoped as the signature (`.ai-chip`).
   - `styles/base.css` — already meets the quality floor: `:focus-visible`,
     `prefers-reduced-motion`, and a mobile breakpoint.
   - `styles/components.css`, `styles/header.css`, an externalized `app.js` (~1,665 lines),
     a clean 207-line semantic `index.html` (hero chips, `<details>` disclosures, Board /
     Season-sim tabs), and an `app.py` change mounting `/static`
     (`plan/artifacts/reskin/app.py.static-mount.diff`).

The reskin's problem was **broken behavior** (the override, in its `app.js`), not missing
styling. Its header took a different direction (hero chips + `<details>`), **without** the
sliver banner.

## Decided Direction

- **Strategy:** reset `frontend/index.html` to HEAD (known-good behavior) and re-apply the
  reskin's styling in staged commits. Do **not** carry the reskin's `app.js`; behavior
  comes from HEAD's proven JS, so the reskin's override bug does not follow us.
- **Header/AI element:** keep the Phase 1 **sliver AI banner** (ported from
  `plan/artifacts/phase1-sliver-index.html`), restyled with the reskin's tokens — not the
  reskin's hero-chips/`<details>` header.
- **Design system:** adopt the reskin's `tokens.css` as the source of truth for color,
  type, spacing, radii, and shadows (see "Design System" below).
- **Aggregate stats:** move into a slide-out drawer that mirrors the settings drawer.
- **CSV export:** drop entirely.
- **Git workflow:** one commit per stage on the working branch — no separate PRs.

### Architecture decision (flag for veto)

Keep `frontend/index.html` **monolithic** (one file, `<style>` + inline `<script>`) for the
rebuild, importing the reskin's tokens as `:root` custom properties and adding the Google
Fonts `<link>`. This is the lowest-risk way to "re-apply styling in stages" on HEAD's
proven inline JS, and it makes Stage 6 theming trivial (tokens already live in `:root`).
The reskin's external `styles/*.css` + `app.js` split and the `/static` mount are treated
as an **optional future refactor, out of scope here** to avoid behavior risk and scope
creep. Say the word to adopt the external-file architecture instead.

## Design System (adopted from `plan/artifacts/reskin/styles/tokens.css`)

- **Color:** `--surface #F7F8FA`, `--surface-raised #FFFFFF`, brand greens
  `--green-700 #1B5E20` / `--green-600 #2E7D32` / `--green-500 #43A047` / `--green-50
  #E8F5E9`, text `--text #1A1D1F` / `--text-muted #5F6368`, `--border #E2E5EA`, on-clock
  amber `--on-clock #E8A317` on `--on-clock-bg #FFF8E7`, `--danger #C62828`.
- **Type:** `--font-ui "DM Sans"` (UI), `--font-stats "IBM Plex Sans"` (labels/stats),
  `--font-mono "IBM Plex Mono"` (numeric data). Scale `--text-xs 12` → `--text-lg 20`.
- **Space/Radius/Shadow:** 4–24px spacing scale; radii 6 / 10 / pill; `--shadow-sm/md`.
- **Signature:** the animated rainbow "lava" gradient, scoped to the AI element only
  (`.ai-chip` / the sliver) — the one bold moment; everything else stays quiet.

**frontend-design critique.** Green-on-light is close to a generic default, but it is
justified here by the subject (fantasy-football draft room) and the existing brand green.
Differentiate deliberately with a **"draft-room scoreboard" data treatment**: IBM Plex Mono
tabular numerics for ADP/VORP/points/pick counts, disciplined hairline dividers, amber
"on the clock" state, and the lava sliver as the single signature. Numbering is used only
where order is real (rounds/picks), never as decoration. Boldness spent in one place; the
rest is restrained.

## Guiding Rules

- **Decouple behavior from presentation.** Bind JS only to `id` / `data-action` — never to
  styling classes.
- **Backend is the single source of truth.** The frontend sends actions and renders the
  server's success/error payload; it does not compute action legality locally.
- **No business-rule `disabled` logic.** Elements stay interactive; the server rejects
  invalid actions and the UI surfaces the message.
- **Behavior-preserving stages.** Keep every handler and its `id`/`data-action` binding
  intact; verify the smoke test after each stage.
- **frontend-design floor:** responsive to mobile, visible keyboard focus,
  `prefers-reduced-motion` respected (all three already exist in the reskin's `base.css`
  and must be carried forward).
- **CSS specificity:** watch for type- vs element-based selectors canceling
  padding/margins between sections.

## Regression Smoke Test (the gate for every stage)

Run after each stage; all must pass before committing:

1. Start New Draft renders the board and player table.
2. Draft a player from the table; row updates and board fills.
3. Undo reverts the last pick.
4. Sim Pick and Auto Draft advance the draft (Bot and Policy engines).
5. **Override:** click a board header for a team not on the clock → that team becomes the
   picker (out-of-turn) with clear visual affordance; a subsequent pick is attributed to
   it. (Pick-trade scenario.)
6. Transfer a drafted player between teams.
7. Sim Season returns results.
8. Ask Assistant returns a recommendation; Auto assistant fires on snake turns per scope.
9. Settings drawer and (new) stats drawer open/close, incl. outside-click and keyboard.

## Stages

### Stage 0 — Preserve reference & baseline
- **Done:** both artifacts preserved (`plan/artifacts/`); stash + working tree intact.
- Reset `frontend/index.html` to HEAD as the known-good rebuild base.
- Establish the smoke test above as the per-stage gate (user runs the app to confirm green
  baseline).
- **Deliverable:** recoverable styling references + green HEAD baseline.

### Stage 1 — Adopt the design-system foundation (additive, no behavior change)
- Add the Google Fonts `<link>` (DM Sans, IBM Plex Sans, IBM Plex Mono).
- Port `tokens.css` into `:root`; add base resets, `:focus-visible`,
  `prefers-reduced-motion`, and the mobile breakpoint from the reskin's `base.css`.
- Wire body/surface/panel/resizer to tokens. No structural or JS changes.
- **Deliverable:** token layer + type in place; app looks lightly refreshed, behaves the
  same.

### Stage 2 — Re-introduce the sliver AI banner + settings drawer
- Port the sliver banner and settings drawer markup from
  `plan/artifacts/phase1-sliver-index.html`, styled with the new tokens.
- Keep bindings to `id`/`data-action`; preserve assistant + settings behavior.
- Verify override and assistant flows in the smoke test.
- **Deliverable:** sliver banner back, on the token system, behavior intact.

### Stage 3 — Restyle core surfaces to the reskin's visual language
- Apply tokens across header chrome, the player table (`.data-table` with IBM Plex Mono
  numerics), the draft board, buttons/chips, resizer, modals, and tooltips.
- Keep all handlers and ids; this is the bulk of "re-apply the reskin's styling."
- **Deliverable:** cohesive reskinned UI with all pre-existing behavior.

### Stage 4 — Aggregate stats drawer
- Reintroduce `GET /api/draft/summary` aggregate stats (Total / QB / RB / WR / TE) in a
  slide-out **drawer mirroring the settings drawer** (`data-action="toggle-stats"`, hidden
  panel, outside-click close, `aria-expanded`, keyboard accessible).
- Restore `fetchDraftSummary()` (called from `renderDraftState`) targeting the drawer.
- **Drop the CSV export entirely** (button, listeners, `downloadCsv()`).
- Remove dead `.stats-bar` / `.team-summary*` CSS.
- **Deliverable:** rarely-used aggregate stats tucked away but one click accessible.

### Stage 5 — Action-driven server validation + toasts (blueprint Phase 2) — DONE
- Global `POST` fetch wrapper (`postJson`) expecting `{ success: boolean, message: string }`;
  on failure (or non-200) it routes to a reusable `showNotification(message, type)` toast
  pipeline (fixed `#toast-container`, red error toast, auto-dismiss ~4s). Done.
- Standardized the error response shape via a single `@app.exception_handler(HTTPException)`
  in `src/draft_buddy/web/app.py` returning `{ success: false, message }` for every endpoint
  (pick, undo, transfer, override, simulate, advisor, etc.); success responses still return
  the draft state the frontend re-renders. **Touches backend.** Done.
- Every mutation action (`startNewDraft`, `draftPlayer`, `undoLastPick`, `overrideTeam`,
  `transferDraftedPlayer`, `simulateNextPick`, `autoDraftRest`) and the initial state load
  now surface errors as toasts instead of blocking `alert()` dialogs. Done.
- Verified: `POST /api/draft/pick` with no body returns `400 {success:false,
  message:"Player ID is required"}` (TestClient).
- Note on disabled/guard logic: the override-out-of-turn breakage (blueprint 2.1's concern)
  was already resolved in Stage 2, so the board override has no disabled guard. The remaining
  `disabled` states are kept intentionally: advisor-model availability (server capability, not
  draft state) and the in-flight auto-draft double-submit guard. Null-guards (`if
  (!currentDraftState)`) were retained but now emit toasts. Can strip further if desired.
- **Deliverable:** consistent, resilient action feedback; no silent UI breakage.

### Stage 6 — Theme switcher (blueprint Phase 3) — DONE
- Added `html[data-theme]` blocks (`dark-slate`, `cyberpunk`, `warm-charcoal`) that redefine
  only color/shadow tokens over `:root`; structure, spacing, typography, and the lava gradient
  are untouched. The `--green-50`/`--green-700` accent-tint/accent-text pair is inverted in
  dark themes so paired components (insight tags) keep contrast. Done.
- Added `<select id="theme-select" data-action="switch-theme">` in the **settings drawer**
  (rarely-changed, tucked away). Done.
- Persistence: an inline `<head>` script applies `localStorage.draftBuddyTheme` before first
  paint (no flash); `initThemeSwitcher()` syncs the picker value and, on change, sets
  `data-theme` on `<html>` and persists. `light` clears the attribute (bare `:root`). Done.
- Native `<select>` option lists in the panel now use `--text`/`--surface-raised` so dropdowns
  stay readable on every theme.
- **Deliverable:** persistent, structure-free theming. Pending user visual pass across
  banner, drawers, board, and table.

### Stage 7 — Cohesion & polish (frontend-design critique)
- Confirm responsive/mobile, visible focus, and reduced-motion (lava animation disabled).
- Final self-critique with screenshots; "remove one accessory" for restraint.
- Fix README links pointing at the old root-level plan paths now under `docs/`.
- Remove `plan/artifacts/` scaffolding once the revamp is stable.
- **Deliverable:** functional, cohesive, distinctive UI.

## Resolved Decisions

1. **Integration approach:** reset to HEAD and re-apply the reskin's styling in staged
   commits.
2. **Git workflow:** a sequence of commits on the working branch — no separate PRs.
3. **CSV export:** drop entirely.
4. **Scope:** all three phases of `docs/DRAFT_BUDDY_UI_REVAMP_PLAN.md`, including the
   Stage 5 backend response-shape standardization (`{ success, message }`).
5. **Typography:** no constraints — adopting the reskin's DM Sans / IBM Plex pairing.
6. **Reskin source of truth:** the `stash@{0}` reskin (preserved in `plan/artifacts/reskin/`)
   supplies the design system; its `app.js` is **not** reused.
7. **Header/AI element:** keep the Phase 1 sliver AI banner, restyled with reskin tokens.
8. **File architecture:** monolithic `index.html` for this effort; the reskin's external
   `styles/*.css` + `app.js` split is an optional future refactor (out of scope).
