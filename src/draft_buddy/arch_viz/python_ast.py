"""
Tree-sitter parser setup for Python sources.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree


@lru_cache(maxsize=1)
def get_parser() -> Parser:
    """
    Return a shared Tree-sitter parser configured for Python.

    Returns
    -------
    Parser
        Parser instance for ``tree_sitter_python``.
    """
    language = Language(tspython.language())
    return Parser(language)


def parse_python_bytes(source: bytes) -> Tree:
    """
    Parse Python source bytes into a Tree-sitter syntax tree.

    Parameters
    ----------
    source : bytes
        UTF-8 encoded Python source.

    Returns
    -------
    Tree
        Root tree for the module.
    """
    return get_parser().parse(source)


def parse_python_file(path: Path) -> Tree:
    """
    Read and parse a Python file from disk.

    Parameters
    ----------
    path : Path
        Path to a ``.py`` file.

    Returns
    -------
    Tree
        Parsed syntax tree.
    """
    p = Path(path)
    data = p.read_bytes()
    return parse_python_bytes(data)
