"""
Build a directed dependency graph from an entry file by following internal imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from tree_sitter import Node

from draft_buddy.arch_viz.import_extractor import extract_import_module_targets
from draft_buddy.arch_viz.project_layout import ProjectLayout, default_stdlib_names, is_external_top_level
from draft_buddy.arch_viz.python_ast import parse_python_bytes


@dataclass
class DependencyGraph:
    """
    Directed graph of internal Python files discovered from an entry point.

    Parameters
    ----------
    project_root : Path
        Repository root.
    nodes : set of Path
        Normalized absolute paths to internal ``.py`` files.
    edges : set of tuple of Path
        ``(source_file, target_file)`` directed edges.
    module_parts_by_file : dict
        Map file path -> module parts for that file.
    trees_by_file : dict
        Parsed Tree-sitter trees for each file (bytes not stored; re-parse if needed).
    source_by_file : dict
        Raw source bytes per file for strategies.
    node_origins : dict, optional
        When set (typically on merged graphs), maps each file path to the entry
        point files whose import traversal reached that module.
    """

    project_root: Path
    nodes: Set[Path] = field(default_factory=set)
    edges: Set[Tuple[Path, Path]] = field(default_factory=set)
    module_parts_by_file: Dict[Path, List[str]] = field(default_factory=dict)
    trees_by_file: Dict[Path, Node] = field(default_factory=dict)
    source_by_file: Dict[Path, bytes] = field(default_factory=dict)
    node_origins: Optional[Dict[Path, Set[Path]]] = None


def _internal_tops(layout: ProjectLayout) -> Set[str]:
    tops: Set[str] = {r.name for r in layout.package_roots}
    tops.update(e.name for e in layout.extra_entry_dirs)
    return tops


def _filter_internal_module(
    parts: List[str],
    layout: ProjectLayout,
    stdlib: frozenset[str],
) -> bool:
    """Return True if ``parts`` should be included as an internal dependency."""
    if not parts:
        return False
    tops = _internal_tops(layout)
    if is_external_top_level(parts[0], tops, stdlib):
        return False
    return layout.module_parts_to_path(parts) is not None


def build_dependency_graph(
    entry_file: Path,
    layout: ProjectLayout,
) -> DependencyGraph:
    """
    Traverse imports starting at ``entry_file`` and collect internal file dependencies.

    Parameters
    ----------
    entry_file : Path
        Path to the Python entry module (e.g. ``api/app.py``).
    layout : ProjectLayout
        Project layout and package roots.

    Returns
    -------
    DependencyGraph
        Graph of internal files and import edges.
    """
    stdlib = default_stdlib_names()
    graph = DependencyGraph(project_root=layout.project_root.resolve())
    queue: List[Path] = []
    seen: Set[Path] = set()

    entry_resolved = entry_file.resolve()
    if not entry_resolved.is_file():
        return graph

    queue.append(entry_resolved)

    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        graph.nodes.add(path)

        mod_parts = layout.file_to_module_parts(path)
        if mod_parts:
            graph.module_parts_by_file[path] = mod_parts

        try:
            data = path.read_bytes()
        except OSError:
            continue
        graph.source_by_file[path] = data
        tree = parse_python_bytes(data)
        graph.trees_by_file[path] = tree.root_node

        targets = extract_import_module_targets(tree.root_node, data, mod_parts)
        for parts in targets:
            if not _filter_internal_module(parts, layout, stdlib):
                continue
            dest = layout.module_parts_to_path(parts)
            if dest is None:
                continue
            dest = dest.resolve()
            graph.edges.add((path, dest))
            if dest not in seen:
                queue.append(dest)

    return graph


def merge_dependency_graphs(
    graphs_with_entries: Iterable[Tuple[Path, DependencyGraph]],
) -> DependencyGraph:
    """
    Union several graphs and record which entry points reach each file.

    Parameters
    ----------
    graphs_with_entries : iterable of tuple
        ``(entry_file, graph)`` pairs from :func:`build_dependency_graph`.

    Returns
    -------
    DependencyGraph
        Merged graph with ``node_origins`` populated for overlap visualization.
    """
    pairs = list(graphs_with_entries)
    if not pairs:
        raise ValueError("merge_dependency_graphs requires at least one graph")
    merged = DependencyGraph(project_root=pairs[0][1].project_root)
    node_origins: Dict[Path, Set[Path]] = {}
    for entry, g in pairs:
        entry_r = entry.resolve()
        for n in g.nodes:
            node_origins.setdefault(n, set()).add(entry_r)
        merged.nodes |= g.nodes
        merged.edges |= g.edges
        merged.module_parts_by_file.update(g.module_parts_by_file)
        merged.trees_by_file.update(g.trees_by_file)
        merged.source_by_file.update(g.source_by_file)
    merged.node_origins = node_origins
    return merged
