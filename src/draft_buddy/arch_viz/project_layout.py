"""
Project roots and module path resolution for internal Python packages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class PackageRoot:
    """
    Map a top-level package name to its directory on disk.

    Parameters
    ----------
    name : str
        Importable top-level name (e.g. ``draft_buddy``).
    path : Path
        Directory containing that package (e.g. ``src/draft_buddy``).
    """

    name: str
    path: Path


@dataclass
class ProjectLayout:
    """
    Describe how filesystem paths map to Python modules for this repository.

    Parameters
    ----------
    project_root : Path
        Repository root (directory containing ``pyproject.toml`` or similar).
    package_roots : list of PackageRoot
        Known internal packages (e.g. ``draft_buddy`` under ``src``).
    extra_entry_dirs : list of Path, optional
        Directories that contain loose modules (e.g. ``api/``, ``scripts/``). Each
        directory name is the top-level prefix for modules under it (e.g. ``scripts/train``).
    """

    project_root: Path
    package_roots: List[PackageRoot] = field(default_factory=list)
    extra_entry_dirs: List[Path] = field(default_factory=list)

    def normalize(self, path: Path) -> Path:
        """
        Return ``path`` resolved and relative to the project root when possible.

        Parameters
        ----------
        path : Path
            Filesystem path.

        Returns
        -------
        Path
            Normalized path.
        """
        try:
            return path.resolve().relative_to(self.project_root.resolve())
        except ValueError:
            return path.resolve()

    def file_to_module_parts(self, file_path: Path) -> Optional[List[str]]:
        """
        Convert a ``.py`` file path to dotted module parts, if it is internal.

        Parameters
        ----------
        file_path : Path
            Path to a Python source file.

        Returns
        -------
        list of str or None
            Module parts (e.g. ``['draft_buddy', 'utils', 'data_utils']``), or
            ``None`` if the file is not under any registered root.
        """
        resolved = file_path.resolve()
        for root in self.package_roots:
            base = root.path.resolve()
            try:
                rel = resolved.relative_to(base)
            except ValueError:
                continue
            parts = list(rel.parts)
            if not parts:
                continue
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = Path(parts[-1]).stem
            return [root.name, *parts]

        for extra in self.extra_entry_dirs:
            base = extra.resolve()
            try:
                rel = resolved.relative_to(base)
            except ValueError:
                continue
            name = extra.name
            stem = Path(rel.name).stem if rel.name.endswith(".py") else rel.name
            if rel.name.endswith(".py") and rel.name != "__init__.py":
                return [name, stem]
        return None

    def module_parts_to_path(self, parts: List[str]) -> Optional[Path]:
        """
        Resolve dotted module parts to a ``.py`` file path under this layout.

        Parameters
        ----------
        parts : list of str
            Module parts (non-empty).

        Returns
        -------
        Path or None
            Absolute path to the module file, or ``None`` if unknown.
        """
        if not parts:
            return None
        top = parts[0]
        for root in self.package_roots:
            if root.name != top:
                continue
            rel_parts = parts[1:]
            if not rel_parts:
                candidate = root.path / "__init__.py"
                return candidate if candidate.is_file() else None
            *packages, last = rel_parts
            base = root.path.joinpath(*packages) if packages else root.path
            mod_file = base / f"{last}.py"
            if mod_file.is_file():
                return mod_file
            init_pkg = base / last / "__init__.py"
            if init_pkg.is_file():
                return init_pkg
        for extra in self.extra_entry_dirs:
            if len(parts) >= 2 and parts[0] == extra.name:
                rel = parts[1:]
                if len(rel) == 1:
                    f = (extra / f"{rel[0]}.py").resolve()
                    if f.is_file():
                        return f
        return None


def default_stdlib_names() -> frozenset[str]:
    """
    Return the set of standard-library top-level module names for this interpreter.

    Returns
    -------
    frozenset of str
        Top-level stdlib names (Python 3.10+ ``sys.stdlib_module_names``).
    """
    names = getattr(sys, "stdlib_module_names", None)
    if names is None:
        return frozenset()
    return frozenset(names)


def is_external_top_level(name: str, internal_tops: Iterable[str], stdlib: frozenset[str]) -> bool:
    """
    Return True if ``name`` is not an internal package prefix and not the stdlib.

    Parameters
    ----------
    name : str
        First segment of an import (e.g. ``numpy``).
    internal_tops : iterable of str
        Allowed internal top-level names (e.g. ``draft_buddy``, ``api``).
    stdlib : frozenset of str
        Standard library top-level names.

    Returns
    -------
    bool
        True when the name should be excluded from the internal graph.
    """
    if name in internal_tops:
        return False
    if name in stdlib:
        return True
    return True
