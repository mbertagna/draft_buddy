"""
Context class that renders Mermaid diagrams using an extraction strategy.
"""

from __future__ import annotations

from draft_buddy.arch_viz.dependency_graph import DependencyGraph
from draft_buddy.arch_viz.project_layout import ProjectLayout
from draft_buddy.arch_viz.strategies.base import ExtractionStrategy


class DiagramGenerator:
    """
    Generate Mermaid diagrams from a dependency graph using an injected strategy.

    Parameters
    ----------
    strategy : ExtractionStrategy
        Strategy that defines diagram granularity and formatting.
    """

    def __init__(self, strategy: ExtractionStrategy) -> None:
        self._strategy = strategy

    def render(self, graph: DependencyGraph, layout: ProjectLayout) -> str:
        """
        Produce Mermaid source text for the given graph.

        Parameters
        ----------
        graph : DependencyGraph
            Parsed dependency graph.
        layout : ProjectLayout
            Project layout for path resolution.

        Returns
        -------
        str
            Mermaid diagram body (without markdown code fences).
        """
        return self._strategy.build(graph, layout)

    @property
    def strategy_name(self) -> str:
        """Return the active strategy name."""
        return self._strategy.name
