"""
AST-based architecture visualization for local dependency analysis.

This package provides entry-point tracing, pluggable extraction strategies,
and Mermaid diagram generation.
"""

from draft_buddy.arch_viz.diagram_generator import DiagramGenerator
from draft_buddy.arch_viz.dependency_graph import (
    DependencyGraph,
    build_dependency_graph,
    merge_dependency_graphs,
)
from draft_buddy.arch_viz.project_layout import PackageRoot, ProjectLayout

__all__ = [
    "DiagramGenerator",
    "DependencyGraph",
    "PackageRoot",
    "ProjectLayout",
    "build_dependency_graph",
    "merge_dependency_graphs",
]
