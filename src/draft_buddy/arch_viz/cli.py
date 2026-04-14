"""
CLI for the architecture visualizer (``draft-arch``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Type

from draft_buddy.arch_viz.dependency_graph import (
    DependencyGraph,
    build_dependency_graph,
    merge_dependency_graphs,
)
from draft_buddy.arch_viz.diagram_generator import DiagramGenerator
from draft_buddy.arch_viz.mermaid_utils import entry_path_slug
from draft_buddy.arch_viz.project_layout import PackageRoot, ProjectLayout
from draft_buddy.arch_viz.strategies.base import ExtractionStrategy
from draft_buddy.arch_viz.strategies.class_strategy import ClassStrategy
from draft_buddy.arch_viz.strategies.function_strategy import FunctionStrategy
from draft_buddy.arch_viz.strategies.module_strategy import ModuleStrategy


def _default_layout(project_root: Path) -> ProjectLayout:
    """
    Build a typical layout for this repository (``draft_buddy`` under ``src``).

    Parameters
    ----------
    project_root : Path
        Repository root.

    Returns
    -------
    ProjectLayout
        Layout with ``draft_buddy`` and optional ``api`` / ``scripts`` roots.
    """
    src = project_root / "src"
    draft = src / "draft_buddy"
    api = project_root / "api"
    scripts = project_root / "scripts"
    roots = [PackageRoot(name="draft_buddy", path=draft)]
    extra: List[Path] = []
    if api.is_dir():
        extra.append(api)
    if scripts.is_dir():
        extra.append(scripts)
    return ProjectLayout(project_root=project_root, package_roots=roots, extra_entry_dirs=extra)


def _strategies() -> Dict[str, Type[ExtractionStrategy]]:
    """Return strategy name to class mapping."""
    return {
        "module": ModuleStrategy,
        "class": ClassStrategy,
        "function": FunctionStrategy,
    }


def _default_multi_entries(project_root: Path) -> List[Path]:
    """
    Return known entry scripts that exist on disk (webapp, training, data prep).

    Parameters
    ----------
    project_root : Path
        Repository root.

    Returns
    -------
    list of Path
        Resolved paths to entry ``.py`` files.
    """
    candidates = [
        project_root / "scripts" / "run_webapp.py",
        project_root / "scripts" / "train.py",
        project_root / "scripts" / "generate_projections.py",
    ]
    return [p.resolve() for p in candidates if p.is_file()]


def _resolve_entry(project_root: Path, raw: Path) -> Path:
    """Resolve a user-supplied entry path against the project root."""
    path = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    if not path.is_file():
        raise SystemExit(f"Entry file not found: {path}")
    return path


def _write_mermaid_file(path: Path, body: str) -> None:
    """Write a fenced Mermaid document to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"```mermaid\n{body}\n```\n", encoding="utf-8")


def main() -> None:
    """Parse CLI arguments and print or write Mermaid diagram output."""
    parser = argparse.ArgumentParser(
        description="Build Mermaid architecture diagrams from Python entry point(s).",
    )
    parser.add_argument(
        "--entry",
        "-e",
        action="append",
        default=[],
        dest="entries",
        metavar="PATH",
        help="Python entry file (repeat for multiple). Default: scripts/run_webapp.py",
    )
    parser.add_argument(
        "--all-default-entries",
        action="store_true",
        help=(
            "Use scripts/run_webapp.py, scripts/train.py, and scripts/generate_projections.py "
            "(only files that exist)."
        ),
    )
    parser.add_argument(
        "--project-root",
        "-p",
        type=Path,
        default=None,
        help="Repository root (default: current working directory)",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        choices=("module", "class", "function"),
        default="module",
        help="Diagram granularity: module, class, or function",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write a single diagram to this file (stdout if omitted; not used with --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=None,
        help="Write one diagram per entry plus a merged diagram (see --merged-name)",
    )
    parser.add_argument(
        "--merged-name",
        default="merged",
        help="Basename for the combined diagram when using --output-dir (default: merged)",
    )
    args = parser.parse_args()

    project_root = (args.project_root or Path.cwd()).resolve()

    if args.all_default_entries and args.entries:
        raise SystemExit("Use either --all-default-entries or --entry, not both.")

    if args.all_default_entries:
        entries = _default_multi_entries(project_root)
        if not entries:
            raise SystemExit("No default entry files found under this project root.")
    elif args.entries:
        entries = [_resolve_entry(project_root, e) for e in args.entries]
    else:
        default = project_root / "scripts" / "run_webapp.py"
        if not default.is_file():
            raise SystemExit(
                f"Default entry not found ({default}). Use --entry or --all-default-entries."
            )
        entries = [default.resolve()]

    layout = _default_layout(project_root)
    strat_cls = _strategies()[args.strategy]
    strategy = strat_cls()
    strat_slug = args.strategy

    if args.output_dir and args.output:
        raise SystemExit("Use either --output or --output-dir, not both.")

    out_dir = args.output_dir
    if out_dir is not None:
        out_path = out_dir if out_dir.is_absolute() else (project_root / out_dir)
        out_path = out_path.resolve()
        graphs_with_entries: List[Tuple[Path, DependencyGraph]] = []
        for entry in entries:
            g = build_dependency_graph(entry, layout)
            graphs_with_entries.append((entry, g))
            slug = entry_path_slug(entry, project_root)
            body = DiagramGenerator(strategy).render(g, layout)
            _write_mermaid_file(out_path / f"{slug}_{strat_slug}.mmd", body)

        if len(entries) > 1:
            merged = merge_dependency_graphs(graphs_with_entries)
            body = DiagramGenerator(strategy).render(merged, layout)
            _write_mermaid_file(out_path / f"{args.merged_name}_{strat_slug}.mmd", body)

        print(f"Wrote diagrams under {out_path}", file=sys.stderr)
        return

    if len(entries) > 1:
        graphs_with_entries = [(e, build_dependency_graph(e, layout)) for e in entries]
        graph = merge_dependency_graphs(graphs_with_entries)
    else:
        graph = build_dependency_graph(entries[0], layout)

    body = DiagramGenerator(strategy).render(graph, layout)
    wrapped = f"```mermaid\n{body}\n```\n"
    if args.output:
        args.output.write_text(wrapped, encoding="utf-8")
    else:
        print(wrapped, end="")


if __name__ == "__main__":
    main()
