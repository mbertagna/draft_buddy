"""Export position guide artifacts to JSON and HTML."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from draft_buddy.data.cache_paths import position_guide_output_path
from draft_buddy.rl.position_guide.schemas import POSITION_CODES, PositionGuideFile

POSITION_BAR_COLORS = {
    "QB": "#4a90d9",
    "RB": "#2ecc71",
    "WR": "#e67e22",
    "TE": "#9b59b6",
}


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
        cells.append(
            f"""        <td class="bar-cell {cell_class}">
          <div>{percent}%</div>
          <div class="bar-wrap"><div class="bar" style="width:{percent}%;background:{color}"></div></div>
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
