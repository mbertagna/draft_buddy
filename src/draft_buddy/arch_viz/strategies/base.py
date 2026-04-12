"""
Abstract extraction strategy for diagram generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from draft_buddy.arch_viz.dependency_graph import DependencyGraph
from draft_buddy.arch_viz.project_layout import ProjectLayout


class ExtractionStrategy(ABC):
    """
    Strategy interface for turning a dependency graph into Mermaid diagram text.

    Subclasses implement a specific granularity (module, class, or function).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for CLI (e.g. ``module``)."""
        raise NotImplementedError

    @abstractmethod
    def build(self, graph: DependencyGraph, layout: ProjectLayout) -> str:
        """
        Produce Mermaid diagram source (without markdown fences).

        Parameters
        ----------
        graph : DependencyGraph
            Parsed internal dependency graph.
        layout : ProjectLayout
            Project layout for path labels.

        Returns
        -------
        str
            Mermaid diagram body.
        """
        raise NotImplementedError
