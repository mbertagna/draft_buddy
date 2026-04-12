"""
Extract import dependency targets from a Tree-sitter Python AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from tree_sitter import Node


def _text(source: bytes, node: Node) -> str:
    """Return UTF-8 text for a node."""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _dotted_name_parts(node: Node, source: bytes) -> List[str]:
    """Split a ``dotted_name`` node into identifier segments."""
    if node.type != "dotted_name":
        return []
    parts: List[str] = []
    for c in node.children:
        if c.type == "identifier":
            parts.append(_text(source, c))
    return parts


def _parse_import_prefix(rel_node: Node, source: bytes) -> Tuple[int, Optional[str]]:
    """
    Parse a ``relative_import`` node into dot count and optional suffix.

    Returns
    -------
    tuple
        ``(n_dots, dotted_suffix)`` where ``dotted_suffix`` is the part after the dots.
    """
    n_dots = 0
    suffix: Optional[str] = None
    for c in rel_node.children:
        if c.type == "import_prefix":
            n_dots = len(_text(source, c))
        elif c.type == "dotted_name":
            suffix = _text(source, c)
    return n_dots, suffix


def _resolve_relative_anchor(
    current_module_parts: Sequence[str],
    n_dots: int,
    dotted_suffix: Optional[str],
) -> Optional[List[str]]:
    """
    Resolve the absolute module for a relative ``from`` clause.

    Parameters
    ----------
    current_module_parts : sequence of str
        Parts of the module containing the import.
    n_dots : int
        Number of leading dots in the relative prefix.
    dotted_suffix : str or None
        Optional dotted path after the dots (e.g. ``parent`` in ``..parent``).

    Returns
    -------
    list of str or None
        Absolute module parts for the ``from`` target, or ``None`` if invalid.
    """
    if n_dots < 1:
        return None
    pkg = list(current_module_parts[:-1])
    up = n_dots - 1
    if up > len(pkg):
        return None
    anchor = pkg[:-up] if up else pkg
    if dotted_suffix:
        anchor = anchor + dotted_suffix.split(".")
    return anchor


def _collect_after_import_nodes(node: Node) -> List[Node]:
    """Return tree-sitter children after the ``import`` keyword."""
    out: List[Node] = []
    seen = False
    for c in node.children:
        if c.type == "import":
            seen = True
            continue
        if seen:
            out.append(c)
    return out


def _targets_from_import_from(
    node: Node,
    source: bytes,
    current_module_parts: Optional[Sequence[str]],
) -> List[List[str]]:
    """
    Collect absolute module parts for a single ``import_from_statement`` node.

    Each returned list is a module whose code may be loaded (dependency target).
    """
    out: List[List[str]] = []
    rel: Optional[Node] = None
    for c in node.children:
        if c.type == "relative_import":
            rel = c
            break

    if rel is not None:
        if current_module_parts is None:
            return out
        n_dots, suffix = _parse_import_prefix(rel, source)
        anchor = _resolve_relative_anchor(current_module_parts, n_dots, suffix)
        if anchor is None:
            return out
        if suffix is None:
            for n in _collect_after_import_nodes(node):
                if n.type == "dotted_as_names":
                    for child in n.children:
                        if child.type == "dotted_as_name":
                            for sub in child.children:
                                if sub.type == "dotted_name":
                                    name_parts = _dotted_name_parts(sub, source)
                                    if name_parts:
                                        out.append(anchor + [name_parts[0]])
                elif n.type == "wildcard_import":
                    pass
        else:
            out.append(anchor)
        return out

    module_parts: Optional[List[str]] = None
    for c in node.children:
        if c.type == "dotted_name":
            module_parts = _dotted_name_parts(c, source)
            break
    if not module_parts:
        return out

    imported_names: List[str] = []
    for n in _collect_after_import_nodes(node):
        if n.type == "wildcard_import":
            out.append(module_parts)
            return out
        if n.type == "dotted_as_names":
            for child in n.children:
                if child.type != "dotted_as_name":
                    continue
                for sub in child.children:
                    if sub.type == "dotted_name":
                        ip = _dotted_name_parts(sub, source)
                        if ip:
                            imported_names.append(ip[0])

    if imported_names:
        for name in imported_names:
            out.append(module_parts + [name])
    else:
        out.append(module_parts)
    return out


def extract_import_module_targets(
    root: Node,
    source: bytes,
    current_module_parts: Optional[Sequence[str]],
) -> List[List[str]]:
    """
    Walk a module AST and return all imported module paths as dotted parts.

    Parameters
    ----------
    root : Node
        Root ``module`` node.
    source : bytes
        Source bytes for the file.
    current_module_parts : sequence of str or None
        Module parts for this file, required to resolve relative imports.

    Returns
    -------
    list of list of str
        Each inner list is a module path (e.g. ``['draft_buddy', 'config']``).
    """
    targets: List[List[str]] = []

    def visit(node: Node) -> None:
        if node.type == "import_statement":
            for c in node.children:
                if c.type == "dotted_as_names":
                    for child in c.children:
                        if child.type == "dotted_as_name":
                            for sub in child.children:
                                if sub.type == "dotted_name":
                                    parts = _dotted_name_parts(sub, source)
                                    if parts:
                                        targets.append(parts)
        elif node.type == "import_from_statement":
            targets.extend(_targets_from_import_from(node, source, current_module_parts))
        for c in node.children:
            visit(c)

    visit(root)
    return targets


@dataclass
class ImportBinding:
    """Maps a local name to an imported module path (parts) or alias."""

    local_name: str
    module_parts: List[str]


def extract_name_bindings(root: Node, source: bytes) -> List[ImportBinding]:
    """
    Collect absolute-import bindings for resolving bare names to defining modules.

    ``from pkg import Name`` maps local ``Name`` to module ``pkg`` (where ``Name`` is defined).

    Parameters
    ----------
    root : Node
        Root ``module`` node.
    source : bytes
        Source bytes.

    Returns
    -------
    list of ImportBinding
        Bindings for absolute ``import`` / ``from ... import`` forms (not relative).
    """
    bindings: List[ImportBinding] = []

    def visit(node: Node) -> None:
        if node.type == "import_statement":
            for c in node.children:
                if c.type != "dotted_as_names":
                    continue
                for child in c.children:
                    if child.type != "dotted_as_name":
                        continue
                    mod_parts: List[str] = []
                    local_name = ""
                    for sub in child.children:
                        if sub.type == "dotted_name":
                            mod_parts = _dotted_name_parts(sub, source)
                            local_name = mod_parts[-1] if mod_parts else ""
                        elif sub.type == "as":
                            continue
                        elif sub.type == "identifier" and local_name:
                            # ``as`` branch handled by checking previous sibling
                            pass
                    # Handle ``import a.b as c``
                    subs = list(child.children)
                    for i, sub in enumerate(subs):
                        if sub.type == "as" and i + 1 < len(subs) and subs[i + 1].type == "identifier":
                            local_name = _text(source, subs[i + 1])
                    if mod_parts:
                        bindings.append(ImportBinding(local_name=local_name, module_parts=mod_parts))
        elif node.type == "import_from_statement":
            if any(c.type == "relative_import" for c in node.children):
                for c in node.children:
                    visit(c)
                return
            mod_parts: List[str] = []
            for c in node.children:
                if c.type == "dotted_name":
                    mod_parts = _dotted_name_parts(c, source)
                    break
            if not mod_parts:
                for c in node.children:
                    visit(c)
                return
            after = _collect_after_import_nodes(node)
            for n in after:
                if n.type != "dotted_as_names":
                    continue
                for child in n.children:
                    if child.type != "dotted_as_name":
                        continue
                    local = ""
                    for sub in child.children:
                        if sub.type == "dotted_name":
                            ps = _dotted_name_parts(sub, source)
                            local = ps[0] if ps else ""
                        elif sub.type == "as":
                            continue
                    subs = list(child.children)
                    for i, sub in enumerate(subs):
                        if sub.type == "as" and i + 1 < len(subs) and subs[i + 1].type == "identifier":
                            local = _text(source, subs[i + 1])
                    if local:
                        bindings.append(ImportBinding(local_name=local, module_parts=list(mod_parts)))
        for c in node.children:
            visit(c)

    visit(root)
    return bindings
