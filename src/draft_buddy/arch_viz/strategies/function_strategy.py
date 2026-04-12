"""
Function- and method-level call extraction (micro-level view).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from tree_sitter import Node

from draft_buddy.arch_viz.dependency_graph import DependencyGraph
from draft_buddy.arch_viz.import_extractor import extract_name_bindings
from draft_buddy.arch_viz.mermaid_utils import display_path, merged_entry_suffix, mermaid_id
from draft_buddy.arch_viz.project_layout import ProjectLayout
from draft_buddy.arch_viz.strategies.base import ExtractionStrategy


def _text(source: bytes, node: Node) -> str:
    """Return UTF-8 text for a node."""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _identifier_name(node: Node, source: bytes) -> str:
    """Return the first identifier child text, or empty string."""
    for c in node.children:
        if c.type == "identifier":
            return _text(source, c)
    return ""


def _function_block(node: Node) -> Optional[Node]:
    """Return the ``block`` child of a ``function_definition``."""
    for c in node.children:
        if c.type == "block":
            return c
    return None


def _collect_identifier_callees(node: Node, source: bytes) -> List[str]:
    """Collect callee names from simple ``identifier(...)`` calls in a subtree."""
    out: List[str] = []

    def visit(n: Node) -> None:
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn is None and n.children:
                fn = n.children[0]
            if fn is not None and fn.type == "identifier":
                out.append(_text(source, fn))
        for c in n.children:
            visit(c)

    visit(node)
    return out


def _collect_functions(
    root: Node,
    source: bytes,
    prefix: Tuple[str, ...],
) -> List[Tuple[str, Node]]:
    """
    Collect qualified function names and their definition nodes.

    Parameters
    ----------
    root : Node
        Module or class body subtree.
    source : bytes
        Source bytes.
    prefix : tuple of str
        Outer class names (empty at module level).

    Returns
    -------
    list of tuple
        ``(qualified_name, function_definition_node)``.
    """
    found: List[Tuple[str, Node]] = []

    def unwrap(node: Node) -> Node:
        """Return inner definition from ``decorated_definition``."""
        if node.type == "decorated_definition":
            for c in node.children:
                if c.type in ("function_definition", "class_definition"):
                    return c
        return node

    def visit_block(block: Node) -> None:
        for raw in block.children:
            ch = unwrap(raw)
            if ch.type == "function_definition":
                fname = _identifier_name(ch, source)
                if not fname:
                    continue
                qual = ".".join(prefix + (fname,)) if prefix else fname
                found.append((qual, ch))
            elif ch.type == "class_definition":
                cname = _identifier_name(ch, source)
                if not cname:
                    continue
                for c2 in ch.children:
                    if c2.type == "block":
                        _collect_from_class(c2, prefix + (cname,))
                        break

    def _collect_from_class(block: Node, pref: Tuple[str, ...]) -> None:
        visit_block(block)

    for ch in root.children:
        ch = unwrap(ch)
        if ch.type == "function_definition":
            fname = _identifier_name(ch, source)
            if fname:
                qual = ".".join(prefix + (fname,)) if prefix else fname
                found.append((qual, ch))
        elif ch.type == "class_definition":
            cname = _identifier_name(ch, source)
            if not cname:
                continue
            for c2 in ch.children:
                if c2.type == "block":
                    _collect_from_class(c2, prefix + (cname,))
                    break

    return found


_SKIP_CALLS = frozenset(
    {
        "print",
        "super",
        "isinstance",
        "issubclass",
        "len",
        "range",
        "set",
        "dict",
        "list",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "enumerate",
        "zip",
        "open",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "getattr",
        "setattr",
        "hasattr",
    }
)


class FunctionStrategy(ExtractionStrategy):
    """
    Map functions/methods to simple identifier calls (micro-level view).
    """

    @property
    def name(self) -> str:
        return "function"

    def build(self, graph: DependencyGraph, layout: ProjectLayout) -> str:
        """
        Build a ``flowchart`` of function-to-callee relationships per file.

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
        proj = layout.project_root.resolve()
        lines: List[str] = ["flowchart LR"]
        emitted: Set[str] = set()

        for path in sorted(graph.nodes, key=lambda p: str(p)):
            data = graph.source_by_file.get(path)
            tree = graph.trees_by_file.get(path)
            if data is None or tree is None:
                continue
            disp = display_path(path, proj)
            binds = extract_name_bindings(tree, data)
            binding_map: Dict[str, List[str]] = {b.local_name: b.module_parts for b in binds}

            funcs = _collect_functions(tree, data, ())
            qual_set = set(q for q, _ in funcs)

            def resolve_local(callee: str) -> Optional[str]:
                if callee in qual_set:
                    return callee
                matches = [q for q in qual_set if q.endswith("." + callee)]
                if len(matches) == 1:
                    return matches[0]
                return None

            for qual, fn_node in funcs:
                block = _function_block(fn_node)
                if block is None:
                    continue
                callees = _collect_identifier_callees(block, data)
                caller_id = mermaid_id(f"{disp}::{qual}")
                file_suffix = merged_entry_suffix(graph, path, proj)
                if caller_id not in emitted:
                    lines.append(f'    {caller_id}["{disp}::{qual}{file_suffix}"]')
                    emitted.add(caller_id)
                for callee in callees:
                    if callee in _SKIP_CALLS:
                        continue
                    target_qual = resolve_local(callee)
                    if target_qual is not None:
                        callee_id = mermaid_id(f"{disp}::{target_qual}")
                        label = f"{disp}::{target_qual}{file_suffix}"
                    elif callee in binding_map:
                        callee_id = mermaid_id(f"{disp}::{qual}->{callee}")
                        label = f"import:{callee}"
                    else:
                        continue
                    if callee_id not in emitted:
                        lines.append(f'    {callee_id}["{label}"]')
                        emitted.add(callee_id)
                    lines.append(f"    {caller_id} --> {callee_id}")

        if len(lines) == 1:
            lines.append('    empty["(no function call edges found)"]')
        return "\n".join(lines)
