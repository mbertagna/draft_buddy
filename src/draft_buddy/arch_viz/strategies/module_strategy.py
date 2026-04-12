"""
Module-level (file-to-file) dependency extraction.
"""

from __future__ import annotations

from draft_buddy.arch_viz.dependency_graph import DependencyGraph
from draft_buddy.arch_viz.mermaid_utils import display_path, merged_entry_suffix, mermaid_id
from draft_buddy.arch_viz.project_layout import ProjectLayout
from draft_buddy.arch_viz.strategies.base import ExtractionStrategy


class ModuleStrategy(ExtractionStrategy):
    """
    Map internal import edges between Python files (macro-level view).
    """

    @property
    def name(self) -> str:
        return "module"

    def build(self, graph: DependencyGraph, layout: ProjectLayout) -> str:
        """
        Build a ``flowchart`` diagram of file-to-file import dependencies.

        Parameters
        ----------
        graph : DependencyGraph
            Dependency graph from the entry point.
        layout : ProjectLayout
            Project layout for labels.

        Returns
        -------
        str
            Mermaid ``flowchart`` source.
        """
        root = layout.project_root.resolve()
        lines = ["flowchart LR"]
        seen_nodes: set[str] = set()
        for src, dst in sorted(graph.edges, key=lambda e: (str(e[0]), str(e[1]))):
            ls = display_path(src, root) + merged_entry_suffix(graph, src, root)
            ld = display_path(dst, root) + merged_entry_suffix(graph, dst, root)
            ids = mermaid_id(display_path(src, root))
            idd = mermaid_id(display_path(dst, root))
            if ids not in seen_nodes:
                lines.append(f'    {ids}["{ls}"]')
                seen_nodes.add(ids)
            if idd not in seen_nodes:
                lines.append(f'    {idd}["{ld}"]')
                seen_nodes.add(idd)
            lines.append(f"    {ids} --> {idd}")
        if len(lines) == 1:
            lines.append('    empty["(no internal imports found)"]')
        return "\n".join(lines)
