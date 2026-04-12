"""
Helpers for stable Mermaid identifiers and labels.
"""

from __future__ import annotations

import re
from pathlib import Path

from draft_buddy.arch_viz.dependency_graph import DependencyGraph


def mermaid_id(label: str) -> str:
    """
    Build a safe Mermaid node id from an arbitrary label.

    Parameters
    ----------
    label : str
        Human-readable label or path fragment.

    Returns
    -------
    str
        Identifier safe for Mermaid (letters, digits, underscore).
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", label)
    if cleaned and cleaned[0].isdigit():
        return f"n_{cleaned}"
    return cleaned or "node"


def display_path(path: Path, project_root: Path) -> str:
    """
    Return a short path string relative to ``project_root`` when possible.

    Parameters
    ----------
    path : Path
        Absolute file path.
    project_root : Path
        Project root directory.

    Returns
    -------
    str
        Display string using forward slashes.
    """
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return rel.as_posix()
    except ValueError:
        return path.resolve().as_posix()


def merged_entry_suffix(graph: DependencyGraph, path: Path, project_root: Path) -> str:
    """
    Return a Mermaid-safe suffix listing which entry points reach a file (merged graphs).

    Parameters
    ----------
    graph : DependencyGraph
        Graph, typically from :func:`~draft_buddy.arch_viz.dependency_graph.merge_dependency_graphs`.
    path : Path
        Absolute path to a Python file in the graph.
    project_root : Path
        Repository root for short entry labels.

    Returns
    -------
    str
        Empty string, or ``\\nentries: ...`` for use inside a quoted Mermaid label.
    """
    if not graph.node_origins:
        return ""
    origins = graph.node_origins.get(path.resolve())
    if not origins:
        return ""

    def _short_entry_label(ep: Path) -> str:
        try:
            rel = ep.resolve().relative_to(project_root.resolve())
            return rel.with_suffix("").as_posix()
        except ValueError:
            return ep.name

    tags = ", ".join(sorted(_short_entry_label(e) for e in origins))
    return f"\\nentries: {tags}"


def merged_entry_note_suffix(graph: DependencyGraph, path: Path, project_root: Path) -> str:
    """
    Return extra lines for a Mermaid ``note`` (plain newlines, not escaped ``\\n``).

    Parameters
    ----------
    graph : DependencyGraph
        Graph with ``node_origins`` set.
    path : Path
        File path for the diagram node.
    project_root : Path
        Repository root.

    Returns
    -------
    str
        Empty string, or a string starting with a newline and ``entries: ...``.
    """
    if not graph.node_origins:
        return ""
    origins = graph.node_origins.get(path.resolve())
    if not origins:
        return ""

    def _short_entry_label(ep: Path) -> str:
        try:
            rel = ep.resolve().relative_to(project_root.resolve())
            return rel.with_suffix("").as_posix()
        except ValueError:
            return ep.name

    tags = ", ".join(sorted(_short_entry_label(e) for e in origins))
    return f"\\nentries: {tags}"


def entry_path_slug(entry: Path, project_root: Path) -> str:
    """
    Return a stable, filename-safe slug for an entry path under the project.

    Parameters
    ----------
    entry : Path
        Entry ``.py`` file.
    project_root : Path
        Repository root.

    Returns
    -------
    str
        Slug such as ``api__app`` or ``src__draft_buddy__utils__data_driver``.
    """
    try:
        rel = entry.resolve().relative_to(project_root.resolve())
        stem = rel.with_suffix("").as_posix().replace("/", "__")
        return stem
    except ValueError:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", entry.stem)
