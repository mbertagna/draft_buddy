"""Render pipeline diagnostics to JSON and a self-contained HTML report."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import CDN

from draft_buddy.data.pipeline_diagnostics import (
    PipelineDiagnostics,
    UnmatchedPlayerRow,
    diagnostics_to_dict,
)

CATEGORY_COLORS = {
    "matched": "#2ecc71",
    "gap": "#e74c3c",
    "rookie": "#95a5a6",
    "retired": "#7f8c8d",
}


def write_pipeline_diagnostics_json(diagnostics: PipelineDiagnostics, output_dir: str) -> str:
    """Write diagnostics JSON beside generated player CSVs.

    Parameters
    ----------
    diagnostics : PipelineDiagnostics
        Structured diagnostics payload.
    output_dir : str
        Directory for the JSON file.

    Returns
    -------
    str
        Path to ``pipeline_diagnostics.json``.
    """
    path = os.path.join(output_dir, "pipeline_diagnostics.json")
    Path(path).write_text(
        json.dumps(diagnostics_to_dict(diagnostics), indent=2),
        encoding="utf-8",
    )
    return path


def write_pipeline_report_html(diagnostics: PipelineDiagnostics, output_dir: str) -> str:
    """Write a self-contained HTML diagnostics report.

    Parameters
    ----------
    diagnostics : PipelineDiagnostics
        Structured diagnostics payload.
    output_dir : str
        Directory for the HTML file.

    Returns
    -------
    str
        Path to ``pipeline_report.html``.
    """
    path = os.path.join(output_dir, "pipeline_report.html")
    Path(path).write_text(render_pipeline_report_html(diagnostics), encoding="utf-8")
    return path


def export_pipeline_report(diagnostics: PipelineDiagnostics, output_dir: str) -> tuple[str, str]:
    """Write JSON and HTML pipeline diagnostics artifacts.

    Parameters
    ----------
    diagnostics : PipelineDiagnostics
        Structured diagnostics payload.
    output_dir : str
        Destination directory.

    Returns
    -------
    tuple[str, str]
        ``(json_path, html_path)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = write_pipeline_diagnostics_json(diagnostics, output_dir)
    html_path = write_pipeline_report_html(diagnostics, output_dir)
    return json_path, html_path


def render_pipeline_report_html(diagnostics: PipelineDiagnostics) -> str:
    """Render the full HTML report as a string.

    Parameters
    ----------
    diagnostics : PipelineDiagnostics
        Structured diagnostics payload.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    funnel_script, funnel_div = _funnel_chart_components(diagnostics)
    scan_script, scan_div = _rank_scan_chart_components(diagnostics)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pipeline report — {escape(diagnostics.league_name)} {diagnostics.draft_year}</title>
  {CDN.render_css()}
  <style>
    :root {{
      --bg: #f7f5f1;
      --ink: #1c1b19;
      --muted: #5c5852;
      --card: #ffffff;
      --line: #d9d3c7;
      --accent: #1f4e79;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }}
    h1, h2, h3 {{
      font-family: "IBM Plex Serif", Georgia, serif;
      font-weight: 600;
      margin: 0 0 0.5rem;
    }}
    h1 {{ font-size: 1.9rem; }}
    h2 {{ font-size: 1.35rem; margin-top: 2rem; }}
    h3 {{ font-size: 1.05rem; margin-top: 1.25rem; }}
    .meta {{
      color: var(--muted);
      margin-bottom: 1.5rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      margin: 1rem 0 1.5rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.85rem 1rem;
    }}
    .card .label {{
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card .value {{
      font-size: 1.4rem;
      font-weight: 650;
      margin-top: 0.2rem;
    }}
    .section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 1rem 1.1rem 1.25rem;
      margin-top: 1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.45rem 0.4rem;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    textarea {{
      width: 100%;
      min-height: 8rem;
      box-sizing: border-box;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85rem;
      padding: 0.75rem;
      border: 1px solid var(--line);
      border-radius: 6px;
      resize: vertical;
      background: #fbfaf7;
    }}
    .copy-row {{
      display: flex;
      gap: 0.5rem;
      align-items: center;
      margin: 0.5rem 0 0.75rem;
    }}
    button {{
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 6px;
      padding: 0.45rem 0.85rem;
      cursor: pointer;
      font-size: 0.9rem;
    }}
    button:hover {{ filter: brightness(1.05); }}
    a {{ color: var(--accent); }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .note {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.4rem; }}
  </style>
</head>
<body>
<main>
  <h1>Pipeline diagnostics</h1>
  <p class="meta">
    {escape(diagnostics.league_name)} ({escape(diagnostics.league_id)})
    · draft year {diagnostics.draft_year}
    · {diagnostics.lookback_seasons}-season lookback
    · generated {escape(generated_at)}
  </p>

  <div class="cards">
    {_metric_card("Sleeper directory", diagnostics.stage_counts.sleeper_directory)}
    {_metric_card("Catalog", diagnostics.stage_counts.catalog)}
    {_metric_card("GSIS resolved", diagnostics.stage_counts.gsis_resolved)}
    {_metric_card("nflverse matched", diagnostics.stage_counts.nflverse_matched)}
    {_metric_card("Rookie projected", diagnostics.stage_counts.rookie_projected)}
  </div>

  <div class="section">
    <h2>Stage funnel</h2>
    {funnel_div}
  </div>

  <div class="section">
    <h2>Sleeper rank scan</h2>
    <p>{escape(diagnostics.nflverse_match.summary_line)}</p>
    {scan_div}
    <h3>nflverse gaps (copyable)</h3>
    {_copyable_player_section("nflverse-gaps", diagnostics.nflverse_match.gap_players)}
  </div>

  <div class="section">
    <h2>Draft-relevant missing veterans</h2>
    <p class="note">
      {diagnostics.draft_relevant_missing.total_missing_veterans} total missing veterans;
      showing search_rank &lt; {diagnostics.draft_relevant_missing.max_search_rank}.
    </p>
    {_copyable_player_section(
        "draft-relevant-missing",
        diagnostics.draft_relevant_missing.players,
    )}
  </div>

  <div class="section">
    <h2>ADP merge</h2>
    <div class="cards">
      {_metric_card("ADP total", diagnostics.adp_match.total_adp_players)}
      {_metric_card("Matched", diagnostics.adp_match.matched_count)}
      {_metric_card("Borderline", diagnostics.adp_match.borderline_count)}
      {_metric_card("Unmatched", diagnostics.adp_match.unmatched_count)}
      {_metric_card("Unmatched DST", diagnostics.adp_match.unmatched_dst_count)}
    </div>
    <h3>Unmatched skill players (copyable)</h3>
    {_copyable_player_section("adp-skill-unmatched", diagnostics.adp_match.skill_unmatched)}
  </div>
</main>
{CDN.render_js()}
{funnel_script}
{scan_script}
<script>
function copyTextarea(id) {{
  const area = document.getElementById(id);
  if (!area) return;
  area.select();
  navigator.clipboard.writeText(area.value);
}}
</script>
</body>
</html>
"""


def _metric_card(label: str, value: int) -> str:
    """Render one summary metric card."""
    return (
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{value}</div></div>'
    )


def _copyable_player_section(section_id: str, players: tuple[UnmatchedPlayerRow, ...]) -> str:
    """Render a table, textarea, and copy button for unmatched players."""
    if not players:
        return '<p class="empty">None.</p>'

    textarea_id = f"{section_id}-copy"
    lines = "\n".join(player.copy_line for player in players)
    rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(player.name)}</td>"
            f"<td>{escape(player.position)}</td>"
            f"<td>{escape(player.detail)}</td>"
            f"<td>{escape(player.reason)}</td>"
            f'<td><a href="{escape(player.google_url)}" target="_blank" rel="noopener">Google</a></td>'
            "</tr>"
        )
        for player in players
    )
    return f"""
<div class="copy-row">
  <button type="button" onclick="copyTextarea('{textarea_id}')">Copy list</button>
  <span class="note">{len(players)} players</span>
</div>
<textarea id="{textarea_id}" readonly>{escape(lines)}</textarea>
<table>
  <thead>
    <tr><th>Name</th><th>Pos</th><th>Detail</th><th>Reason</th><th>Search</th></tr>
  </thead>
  <tbody>
{rows}
  </tbody>
</table>
"""


def _funnel_chart_components(diagnostics: PipelineDiagnostics) -> tuple[str, str]:
    """Build Bokeh script/div for the stage funnel chart."""
    stages = [
        ("Sleeper directory", diagnostics.stage_counts.sleeper_directory),
        ("Catalog", diagnostics.stage_counts.catalog),
        ("GSIS resolved", diagnostics.stage_counts.gsis_resolved),
        ("nflverse matched", diagnostics.stage_counts.nflverse_matched),
        ("Rookie projected", diagnostics.stage_counts.rookie_projected),
    ]
    labels = [label for label, _ in stages]
    values = [value for _, value in stages]
    source = ColumnDataSource(data={"stage": labels, "count": values})
    plot = figure(
        y_range=list(reversed(labels)),
        height=280,
        width=900,
        toolbar_location=None,
        tools="",
        title="Players retained by stage",
    )
    plot.hbar(y="stage", right="count", height=0.6, source=source, color="#1f4e79")
    plot.xaxis.axis_label = "Players"
    plot.yaxis.axis_label = None
    plot.add_tools(HoverTool(tooltips=[("Stage", "@stage"), ("Count", "@count")]))
    return components(plot)


def _rank_scan_chart_components(diagnostics: PipelineDiagnostics) -> tuple[str, str]:
    """Build Bokeh script/div for the Sleeper rank-scan strip."""
    rows = diagnostics.nflverse_match.scan_rows
    if not rows:
        return "", '<p class="empty">No rank-scan rows available.</p>'

    ranks = [row.search_rank for row in rows]
    names = [row.name for row in rows]
    positions = [row.position for row in rows]
    categories = [row.category for row in rows]
    colors = [CATEGORY_COLORS.get(row.category, "#333333") for row in rows]
    source = ColumnDataSource(
        data={
            "rank": ranks,
            "name": names,
            "position": positions,
            "category": categories,
            "color": colors,
            "y": [0] * len(rows),
        }
    )
    plot = figure(
        height=160,
        width=900,
        toolbar_location="above",
        tools="pan,xwheel_zoom,reset,save",
        active_scroll="xwheel_zoom",
        title="Sleeper search_rank scan",
        x_axis_label="search_rank",
    )
    plot.scatter(x="rank", y="y", size=8, color="color", source=source, alpha=0.85)
    plot.yaxis.visible = False
    plot.y_range.start = -1
    plot.y_range.end = 1
    plot.add_tools(
        HoverTool(
            tooltips=[
                ("Rank", "@rank"),
                ("Name", "@name"),
                ("Pos", "@position"),
                ("Category", "@category"),
            ]
        )
    )
    return components(plot)
