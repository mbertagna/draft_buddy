"""Export position guide artifacts to JSON and HTML."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from draft_buddy.data.cache_paths import (
    model_adp_by_position_output_path,
    model_adp_output_path,
    position_guide_output_path,
)
from draft_buddy.rl.position_guide.schemas import (
    POSITION_CODES,
    ModelAdpByPositionFile,
    ModelAdpFile,
    PositionGuideFile,
    TopPlayerEntry,
)

POSITION_BAR_COLORS = {
    "QB": "#4a90d9",
    "RB": "#2ecc71",
    "WR": "#e67e22",
    "TE": "#9b59b6",
}


def _pool_meta_suffix(
    prune_inactive: bool,
    limit_adp: int | None,
    draft_pool_size: int | None,
) -> str:
    """Return a short HTML meta fragment describing draft-pool restrictions."""
    parts: list[str] = []
    if prune_inactive:
        parts.append("prune-inactive")
    if limit_adp is not None:
        parts.append(f"limit-adp={limit_adp}")
    if draft_pool_size is not None:
        parts.append(f"pool={draft_pool_size}")
    if not parts:
        return ""
    return " · " + " · ".join(parts)


def _adp_table_styles() -> str:
    """Return shared CSS for interactive ADP ranking tables with draft checkboxes."""
    return """
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 1rem;
      color: #1a1a1a;
      background: #fafafa;
    }
    h1 { font-size: 1.25rem; margin-bottom: 0.25rem; }
    .meta { color: #555; font-size: 0.85rem; margin-bottom: 0.75rem; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.75rem 1rem;
      margin-bottom: 0.75rem;
      font-size: 0.85rem;
    }
    .toolbar label {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .toolbar select,
    .toolbar button {
      font: inherit;
      padding: 0.25rem 0.5rem;
    }
    .toolbar button {
      cursor: pointer;
      border: 1px solid #ccc;
      border-radius: 4px;
      background: #fff;
    }
    .toolbar button:hover { background: #f0f0f0; }
    .print-hint { color: #666; font-size: 0.8rem; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    th, td {
      border: 1px solid #ddd;
      padding: 0.4rem 0.5rem;
      text-align: center;
      font-size: 0.85rem;
    }
    th { background: #2c3e50; color: #fff; }
    th.sortable {
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }
    th.sortable:hover { background: #3d566e; }
    th.sortable::after {
      content: "";
      display: inline-block;
      width: 0.75rem;
      margin-left: 0.2rem;
      opacity: 0.45;
    }
    th.sortable.sort-asc::after { content: "▲"; opacity: 1; }
    th.sortable.sort-desc::after { content: "▼"; opacity: 1; }
    td.name-cell { text-align: left; white-space: nowrap; }
    .check-cell { width: 1.5rem; }
    input[type="checkbox"] {
      width: 1rem;
      height: 1rem;
      cursor: pointer;
    }
    tr.drafted td.name-cell {
      text-decoration: line-through;
      color: #888;
    }
    tr[hidden] { display: none; }
    tbody tr { page-break-inside: avoid; }
  @media print {
    body { margin: 0.5rem; background: #fff; }
    .no-print { display: none !important; }
    table { box-shadow: none; }
    input[type="checkbox"] {
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
  }"""


def _interactive_adp_table_script(storage_key: str) -> str:
    """Return client-side JS for filter, sort, and persistent draft checkboxes."""
    safe_key = json.dumps(storage_key)
    return f"""
    (() => {{
      const storageKey = {safe_key};
      const tbody = document.getElementById("adp-body");
      const positionFilter = document.getElementById("position-filter");
      const clearDraftedButton = document.getElementById("clear-drafted");
      const sortableHeaders = Array.from(document.querySelectorAll("th.sortable"));
      let sortColumn = "rank";
      let sortDirection = "asc";

      function loadDrafted() {{
        try {{
          return JSON.parse(localStorage.getItem(storageKey)) || {{}};
        }} catch (_error) {{
          return {{}};
        }}
      }}

      function saveDrafted(drafted) {{
        localStorage.setItem(storageKey, JSON.stringify(drafted));
      }}

      function rowValue(row, column) {{
        return row.dataset[column] ?? "";
      }}

      function compareRows(left, right) {{
        const numericColumns = new Set([
          "rank",
          "modelAdp",
          "marketAdp",
          "stdDev",
          "draftRate",
        ]);
        const leftValue = rowValue(left, sortColumn);
        const rightValue = rowValue(right, sortColumn);
        let result;
        if (numericColumns.has(sortColumn)) {{
          const leftNumber = Number(leftValue);
          const rightNumber = Number(rightValue);
          const leftFinite = Number.isFinite(leftNumber);
          const rightFinite = Number.isFinite(rightNumber);
          if (!leftFinite && !rightFinite) {{
            result = 0;
          }} else if (!leftFinite) {{
            result = 1;
          }} else if (!rightFinite) {{
            result = -1;
          }} else {{
            result = leftNumber - rightNumber;
          }}
        }} else {{
          result = leftValue.localeCompare(rightValue, undefined, {{ sensitivity: "base" }});
        }}
        return sortDirection === "asc" ? result : -result;
      }}

      function updateSortIndicators() {{
        sortableHeaders.forEach((header) => {{
          header.classList.remove("sort-asc", "sort-desc");
          if (header.dataset.sort === sortColumn) {{
            header.classList.add(sortDirection === "asc" ? "sort-asc" : "sort-desc");
          }}
        }});
      }}

      function applyView() {{
        const filter = positionFilter.value;
        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.forEach((row) => {{
          const matchesPosition = filter === "ALL" || row.dataset.position === filter;
          row.hidden = !matchesPosition;
        }});
        const visibleRows = rows.filter((row) => !row.hidden);
        visibleRows.sort(compareRows);
        visibleRows.forEach((row, index) => {{
          const rankCell = row.querySelector(".rank-cell");
          if (rankCell) {{
            rankCell.textContent = String(index + 1);
          }}
          tbody.appendChild(row);
        }});
        rows.filter((row) => row.hidden).forEach((row) => tbody.appendChild(row));
      }}

      function bindCheckbox(row) {{
        const checkbox = row.querySelector('input[type="checkbox"]');
        if (!checkbox) {{
          return;
        }}
        const playerId = row.dataset.playerId;
        const drafted = loadDrafted();
        checkbox.checked = Boolean(drafted[playerId]);
        row.classList.toggle("drafted", checkbox.checked);
        checkbox.addEventListener("change", () => {{
          const state = loadDrafted();
          if (checkbox.checked) {{
            state[playerId] = true;
          }} else {{
            delete state[playerId];
          }}
          saveDrafted(state);
          row.classList.toggle("drafted", checkbox.checked);
        }});
      }}

      sortableHeaders.forEach((header) => {{
        header.addEventListener("click", () => {{
          const column = header.dataset.sort;
          if (sortColumn === column) {{
            sortDirection = sortDirection === "asc" ? "desc" : "asc";
          }} else {{
            sortColumn = column;
            sortDirection = column === "name" || column === "position" ? "asc" : "asc";
          }}
          updateSortIndicators();
          applyView();
        }});
      }});

      positionFilter.addEventListener("change", applyView);
      clearDraftedButton.addEventListener("click", () => {{
        localStorage.removeItem(storageKey);
        tbody.querySelectorAll("tr").forEach((row) => {{
          const checkbox = row.querySelector('input[type="checkbox"]');
          if (checkbox) {{
            checkbox.checked = false;
          }}
          row.classList.remove("drafted");
        }});
      }});

      tbody.querySelectorAll("tr").forEach(bindCheckbox);
      updateSortIndicators();
      applyView();
    }})();"""


def _render_adp_checkbox_cell(player_id: int, player_name: str) -> str:
    """Render a draft-checkbox cell for one player row.

    Parameters
    ----------
    player_id : int
        Player identifier used for the checkbox element id.
    player_name : str
        Player display name for the checkbox aria-label.

    Returns
    -------
    str
        HTML table cell containing a checkbox input.
    """
    label = escape(f"Drafted {player_name}")
    return (
        f'        <td class="check-cell">'
        f'<input type="checkbox" id="player-{player_id}" aria-label="{label}"></td>'
    )


def build_model_adp_by_position(model_adp: ModelAdpFile) -> ModelAdpByPositionFile:
    """Group model ADP players by position and rank within each position.

    Parameters
    ----------
    model_adp : ModelAdpFile
        League-wide model ADP export from simulation.

    Returns
    -------
    ModelAdpByPositionFile
        Same metadata as ``model_adp`` with players grouped and sorted by
        ascending model ADP within each position.
    """
    positions = {
        position: sorted(
            (entry for entry in model_adp.players if entry.position == position),
            key=lambda entry: entry.model_adp,
        )
        for position in POSITION_CODES
    }
    return ModelAdpByPositionFile(
        generated_at=model_adp.generated_at,
        draft_year=model_adp.draft_year,
        num_teams=model_adp.num_teams,
        simulations=model_adp.simulations,
        checkpoint_path=model_adp.checkpoint_path,
        checkpoint_episode=model_adp.checkpoint_episode,
        player_data_csv=model_adp.player_data_csv,
        temperature=model_adp.temperature,
        prune_inactive=model_adp.prune_inactive,
        limit_adp=model_adp.limit_adp,
        draft_pool_size=model_adp.draft_pool_size,
        positions=positions,
    )


def export_position_guide(
    guide: PositionGuideFile,
    data_root: str,
) -> tuple[str, str]:
    """Write JSON and HTML exports for a position guide.

    Parameters
    ----------
    guide : PositionGuideFile
        Guide content to export.
    data_root : str
        Root data directory for output paths.

    Returns
    -------
    tuple[str, str]
        Paths to the JSON and HTML files.
    """
    generated_at = guide.generated_at
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=datetime.now().astimezone().tzinfo)

    json_path = position_guide_output_path(
        data_root=data_root,
        num_teams=guide.num_teams,
        slot=guide.draft_slot,
        year=guide.draft_year,
        generated_at=generated_at,
        ext="json",
    )
    html_path = position_guide_output_path(
        data_root=data_root,
        num_teams=guide.num_teams,
        slot=guide.draft_slot,
        year=guide.draft_year,
        generated_at=generated_at,
        ext="html",
    )

    json_payload = guide.model_dump(mode="json")
    Path(json_path).write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )
    Path(html_path).write_text(
        render_position_guide_html(guide),
        encoding="utf-8",
    )
    return json_path, html_path


def render_position_guide_html(guide: PositionGuideFile) -> str:
    """Render a self-contained printable HTML cheat sheet.

    Parameters
    ----------
    guide : PositionGuideFile
        Guide content to render.

    Returns
    -------
    str
        HTML document as a string.
    """
    title = (
        f"Position Guide — {guide.num_teams}-team slot {guide.draft_slot} "
        f"({guide.draft_year})"
    )
    header_meta = (
        f"Checkpoint episode {guide.checkpoint_episode} · "
        f"{guide.simulations:,} simulations · "
        f"temperature {guide.temperature:g} · "
        f"{guide.generation_mode}"
        f"{_pool_meta_suffix(guide.prune_inactive, guide.limit_adp, guide.draft_pool_size)} · "
        f"Generated {guide.generated_at.isoformat()}"
    )
    rows_html = "\n".join(_render_pick_row(pick) for pick in guide.picks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 1rem;
      color: #1a1a1a;
      background: #fafafa;
    }}
    h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}
    .meta {{ color: #555; font-size: 0.85rem; margin-bottom: 1rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 0.5rem 0.4rem;
      text-align: center;
      font-size: 0.85rem;
    }}
    th {{ background: #2c3e50; color: #fff; }}
    td.pick-info {{ text-align: left; white-space: nowrap; }}
    .bar-cell {{ min-width: 4rem; }}
    .bar-wrap {{
      background: #eee;
      border-radius: 3px;
      height: 1rem;
      overflow: hidden;
      margin-top: 0.15rem;
    }}
    .bar {{
      height: 100%;
      border-radius: 3px;
    }}
    tr.top-row td.pos-top {{
      font-weight: 700;
      background: #fffde7;
    }}
    ul.top-players {{
      list-style: none;
      margin: 0.25rem 0 0;
      padding: 0;
      font-size: 0.7rem;
      font-weight: 400;
      color: #444;
      text-align: left;
    }}
    ul.top-players li {{
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="meta">{escape(header_meta)}</p>
  <table>
    <thead>
      <tr>
        <th>Your Pick</th>
        <th>Round</th>
        <th>Overall</th>
        <th>Top</th>
        <th>QB</th>
        <th>RB</th>
        <th>WR</th>
        <th>TE</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
</body>
</html>
"""


def _render_pick_row(pick) -> str:
    """Render one table row for a pick entry.

    Parameters
    ----------
    pick : PositionGuidePick
        Pick row to render.

    Returns
    -------
    str
        HTML table row.
    """
    positions = pick.positions.model_dump()
    cells = []
    for position in POSITION_CODES:
        probability = positions[position]
        percent = int(round(probability * 100))
        color = POSITION_BAR_COLORS[position]
        is_top = position == pick.top_position
        cell_class = "pos-top" if is_top else ""
        top_players_html = _render_top_players(pick.top_players.get(position, []))
        cells.append(
            f"""        <td class="bar-cell {cell_class}">
          <div>{percent}%</div>
          <div class="bar-wrap"><div class="bar" style="width:{percent}%;background:{color}"></div></div>
          {top_players_html}
        </td>"""
        )
    row_class = "top-row"
    return f"""      <tr class="{row_class}">
        <td class="pick-info">#{pick.user_pick_index}</td>
        <td>{pick.round}</td>
        <td>{pick.overall_pick_number}</td>
        <td><strong>{escape(pick.top_position)}</strong></td>
{chr(10).join(cells)}
      </tr>"""


def export_model_adp(model_adp: ModelAdpFile, data_root: str) -> tuple[str, str]:
    """Write JSON and HTML exports for a model-derived ADP ranking.

    Parameters
    ----------
    model_adp : ModelAdpFile
        Model-derived ADP content to export.
    data_root : str
        Root data directory for output paths.

    Returns
    -------
    tuple[str, str]
        Paths to the JSON and HTML files.
    """
    generated_at = model_adp.generated_at
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=datetime.now().astimezone().tzinfo)

    json_path = model_adp_output_path(
        data_root=data_root,
        num_teams=model_adp.num_teams,
        year=model_adp.draft_year,
        generated_at=generated_at,
        ext="json",
    )
    html_path = model_adp_output_path(
        data_root=data_root,
        num_teams=model_adp.num_teams,
        year=model_adp.draft_year,
        generated_at=generated_at,
        ext="html",
    )

    json_payload = model_adp.model_dump(mode="json")
    Path(json_path).write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    Path(html_path).write_text(render_model_adp_html(model_adp), encoding="utf-8")
    return json_path, html_path


def render_model_adp_html(model_adp: ModelAdpFile) -> str:
    """Render an interactive, printable HTML model-ADP cheat sheet.

    Parameters
    ----------
    model_adp : ModelAdpFile
        Model-derived ADP content to render.

    Returns
    -------
    str
        HTML document as a string.
    """
    title = f"Model ADP — {model_adp.num_teams}-team league ({model_adp.draft_year})"
    header_meta = _model_adp_header_meta(model_adp)
    storage_key = f"draft-buddy-adp-{model_adp.generated_at.isoformat()}"
    return _render_model_adp_document(title, header_meta, model_adp.players, storage_key)


def _model_adp_header_meta(model_adp: ModelAdpFile | ModelAdpByPositionFile) -> str:
    """Return shared metadata text for model ADP HTML exports."""
    return (
        f"Checkpoint episode {model_adp.checkpoint_episode} · "
        f"{model_adp.simulations:,} simulations · "
        f"temperature {model_adp.temperature:g}"
        f"{_pool_meta_suffix(model_adp.prune_inactive, model_adp.limit_adp, model_adp.draft_pool_size)} · "
        f"Generated {model_adp.generated_at.isoformat()}"
    )


def _render_model_adp_document(
    title: str,
    header_meta: str,
    entries: list,
    storage_key: str,
) -> str:
    """Render a complete interactive HTML document for a model ADP ranking table.

    Parameters
    ----------
    title : str
        Document title and page heading.
    header_meta : str
        Subheading metadata line.
    entries : list
        Ordered ``ModelAdpEntry`` rows to render.
    storage_key : str
        localStorage key used to persist drafted-player checkboxes.

    Returns
    -------
    str
        HTML document as a string.
    """
    rows_html = "\n".join(
        _render_model_adp_row(entry, rank) for rank, entry in enumerate(entries, start=1)
    )
    position_options = "\n".join(
        f'      <option value="{position}">{position}</option>' for position in POSITION_CODES
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{_adp_table_styles()}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="meta">{escape(header_meta)}</p>
  <div class="toolbar no-print">
    <label for="position-filter">Position
      <select id="position-filter">
        <option value="ALL">All</option>
{position_options}
      </select>
    </label>
    <button type="button" id="clear-drafted">Clear drafted</button>
    <span class="print-hint">Print uses the current filter and sort. Drafted checkboxes persist while filtering.</span>
  </div>
  <table id="adp-table">
    <thead>
      <tr>
        <th></th>
        <th class="sortable" data-sort="rank">#</th>
        <th class="sortable" data-sort="name">Player</th>
        <th class="sortable" data-sort="position">Pos</th>
        <th class="sortable" data-sort="modelAdp">Model ADP</th>
        <th class="sortable" data-sort="marketAdp">Market ADP</th>
        <th class="sortable" data-sort="stdDev">Std Dev</th>
        <th class="sortable" data-sort="draftRate">Draft Rate</th>
      </tr>
    </thead>
    <tbody id="adp-body">
{rows_html}
    </tbody>
  </table>
  <script>{_interactive_adp_table_script(storage_key)}</script>
</body>
</html>
"""


def _render_model_adp_row(entry, rank: int) -> str:
    """Render one table row for a model-ADP entry."""
    market_adp_value = "" if entry.market_adp is None else f"{entry.market_adp}"
    market_adp_display = "-" if entry.market_adp is None else f"{entry.market_adp:.1f}"
    checkbox_cell = _render_adp_checkbox_cell(entry.player_id, entry.name)
    return f"""      <tr
        data-player-id="{entry.player_id}"
        data-rank="{rank}"
        data-name="{escape(entry.name)}"
        data-position="{escape(entry.position)}"
        data-model-adp="{entry.model_adp}"
        data-market-adp="{market_adp_value}"
        data-std-dev="{entry.std_dev}"
        data-draft-rate="{entry.draft_rate}">
{checkbox_cell}
        <td class="rank-cell">{rank}</td>
        <td class="name-cell">{escape(entry.name)}</td>
        <td>{escape(entry.position)}</td>
        <td>{entry.model_adp:.1f}</td>
        <td>{market_adp_display}</td>
        <td>{entry.std_dev:.1f}</td>
        <td>{entry.draft_rate * 100:.0f}%</td>
      </tr>"""


def export_model_adp_by_position(
    model_adp_by_position: ModelAdpByPositionFile,
    data_root: str,
) -> str:
    """Write a JSON export for model ADP grouped by position.

    Parameters
    ----------
    model_adp_by_position : ModelAdpByPositionFile
        Model ADP grouped by position.
    data_root : str
        Root data directory for output paths.

    Returns
    -------
    str
        Path to the combined JSON file.
    """
    generated_at = model_adp_by_position.generated_at
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=datetime.now().astimezone().tzinfo)

    json_path = model_adp_by_position_output_path(
        data_root=data_root,
        num_teams=model_adp_by_position.num_teams,
        year=model_adp_by_position.draft_year,
        generated_at=generated_at,
        ext="json",
    )
    json_payload = model_adp_by_position.model_dump(mode="json")
    Path(json_path).write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return json_path


def _render_top_players(entries: list[TopPlayerEntry]) -> str:
    """Render the small top-5 drafted-player list under a position bar.

    Parameters
    ----------
    entries : list[TopPlayerEntry]
        Up to five most-frequently-drafted players for this position and
        pick, ordered by frequency.

    Returns
    -------
    str
        HTML snippet, empty when there are no entries.
    """
    if not entries:
        return ""
    items = "".join(
        f"<li>{escape(entry.name)} ({entry.share * 100:.0f}%)</li>" for entry in entries
    )
    return f'<ul class="top-players">{items}</ul>'
