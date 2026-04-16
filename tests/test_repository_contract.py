"""Repository-level contract tests for the canonical cleanup."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_IMPORT_PATTERNS = [
    "draft_buddy." + "domain",
    "draft_buddy." + "logic",
    "draft_buddy." + "data_pipeline",
    "draft_buddy." + "draft_env",
]


def test_legacy_packages_are_removed() -> None:
    """Verify deleted top-level packages are absent from the source tree."""
    legacy_dirs = [
        REPO_ROOT / "src" / "draft_buddy" / "domain",
        REPO_ROOT / "src" / "draft_buddy" / "logic",
        REPO_ROOT / "src" / "draft_buddy" / "data_pipeline",
        REPO_ROOT / "src" / "draft_buddy" / "draft_env",
    ]

    assert all(not directory.exists() for directory in legacy_dirs)


def test_repository_contains_no_legacy_import_paths() -> None:
    """Verify source, scripts, tests, and README reference only canonical package names."""
    file_paths = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").rglob("*.py")) + list((REPO_ROOT / "tests").rglob("*.py")) + [REPO_ROOT / "README.md"]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in file_paths)

    assert all(pattern not in contents for pattern in LEGACY_IMPORT_PATTERNS)


def test_readme_package_tree_matches_canonical_layout() -> None:
    """Verify README documents only the canonical package buckets."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert all(label in readme for label in ["core/", "data/", "simulator/", "rl/", "web/", "arch_viz/"])
