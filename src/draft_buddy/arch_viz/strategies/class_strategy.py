"""
Class-level extraction: definitions, inheritance, and cross-module resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from tree_sitter import Node

from draft_buddy.arch_viz.dependency_graph import DependencyGraph
from draft_buddy.arch_viz.import_extractor import extract_name_bindings
from draft_buddy.arch_viz.mermaid_utils import display_path, mermaid_id, merged_entry_note_suffix
from draft_buddy.arch_viz.project_layout import ProjectLayout
from draft_buddy.arch_viz.strategies.base import ExtractionStrategy


def _text(source: bytes, node: Node) -> str:
    """Return UTF-8 text for a node."""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _base_name_from_node(node: Node, source: bytes) -> str:
    """
    Return a simple name for a base expression (identifier or dotted attribute tail).

    Parameters
    ----------
    node : Node
        ``identifier`` or ``attribute`` base expression.
    source : bytes
        Source bytes.

    Returns
    -------
    str
        Unqualified name used for import / local resolution.
    """
    if node.type == "identifier":
        return _text(source, node)
    if node.type == "attribute":
        for c in reversed(node.children):
            if c.type == "identifier":
                return _text(source, c)
    return ""


@dataclass(frozen=True)
class _ClassInfo:
    """Parsed class metadata for diagram edges."""

    file_display: str
    name: str
    bases: Tuple[str, ...]


def _iter_class_infos(root: Node, source: bytes, file_display: str) -> List[_ClassInfo]:
    """Collect class names and base identifiers from a module AST."""
    out: List[_ClassInfo] = []

    def visit(node: Node) -> None:
        if node.type == "class_definition":
            name = ""
            bases: List[str] = []
            for c in node.children:
                if c.type == "identifier" and not name:
                    name = _text(source, c)
                elif c.type == "argument_list":
                    for a in c.children:
                        if a.type in ("identifier", "attribute"):
                            bn = _base_name_from_node(a, source)
                            if bn:
                                bases.append(bn)
            if name:
                out.append(_ClassInfo(file_display=file_display, name=name, bases=tuple(bases)))
        for c in node.children:
            visit(c)

    visit(root)
    return out


def _builtin_bases() -> Set[str]:
    """Names that are not resolved as internal classes."""
    return {
        "object",
        "ABC",
        "ABCMeta",
        "Generic",
        "Protocol",
        "TypedDict",
        "NamedTuple",
        "Enum",
        "IntEnum",
        "Exception",
        "BaseException",
    }


class ClassStrategy(ExtractionStrategy):
    """
    Map classes and inheritance within the reachable graph (mid-level view).
    """

    @property
    def name(self) -> str:
        return "class"

    def build(self, graph: DependencyGraph, layout: ProjectLayout) -> str:
        """
        Build a ``classDiagram`` for classes and inheritance edges.

        Parameters
        ----------
        graph : DependencyGraph
            Dependency graph from the entry point.
        layout : ProjectLayout
            Project layout for labels.

        Returns
        -------
        str
            Mermaid ``classDiagram`` source.
        """
        root = layout.project_root.resolve()
        classes_by_file: Dict[str, List[_ClassInfo]] = {}
        local_names: Dict[str, Set[str]] = {}

        for path in sorted(graph.nodes, key=lambda p: str(p)):
            data = graph.source_by_file.get(path)
            tree = graph.trees_by_file.get(path)
            if data is None or tree is None:
                continue
            disp = display_path(path, root)
            infos = _iter_class_infos(tree, data, disp)
            classes_by_file[disp] = infos
            local_names[disp] = {c.name for c in infos}

        class_keys: Set[Tuple[str, str]] = set()
        for disp, infos in classes_by_file.items():
            for c in infos:
                class_keys.add((disp, c.name))

        binding_by_file: Dict[str, Dict[str, List[str]]] = {}
        for path in graph.nodes:
            data = graph.source_by_file.get(path)
            tree = graph.trees_by_file.get(path)
            if data is None or tree is None:
                continue
            disp = display_path(path, root)
            binds = extract_name_bindings(tree, data)
            binding_by_file[disp] = {b.local_name: b.module_parts for b in binds}

        builtins = _builtin_bases()
        lines: List[str] = ["classDiagram"]
        class_ids: Dict[Tuple[str, str], str] = {}
        relation_pairs: Set[Tuple[str, str]] = set()

        for disp, infos in sorted(classes_by_file.items()):
            for c in infos:
                uid = mermaid_id(f"{disp}::{c.name}")
                class_ids[(disp, c.name)] = uid

        for path in sorted(graph.nodes, key=lambda p: str(p)):
            disp = display_path(path, root)
            suffix = merged_entry_note_suffix(graph, path, root)
            for c in classes_by_file.get(disp, []):
                uid = class_ids[(disp, c.name)]
                lines.append(f"    class {uid}")
                lines.append(f"    note for {uid} \"{c.name}\\n{disp}{suffix}\"")

        for disp, infos in sorted(classes_by_file.items()):
            for c in infos:
                cid = class_ids[(disp, c.name)]
                for base in c.bases:
                    if base in builtins:
                        continue
                    target_file: str | None = None
                    if base in local_names.get(disp, set()):
                        target_file = disp
                    else:
                        mod_parts = binding_by_file.get(disp, {}).get(base)
                        if mod_parts:
                            p = layout.module_parts_to_path(mod_parts)
                            if p is not None:
                                target_file = display_path(p.resolve(), root)
                    if target_file is None:
                        continue
                    if (target_file, base) not in class_keys:
                        continue
                    bid = class_ids[(target_file, base)]
                    key = (cid, bid)
                    if key in relation_pairs:
                        continue
                    relation_pairs.add(key)
                    lines.append(f"    {cid} --|> {bid}")

        if len(lines) == 1:
            lines.append("    note \"No classes found in reachable modules\"")
        return "\n".join(lines)
