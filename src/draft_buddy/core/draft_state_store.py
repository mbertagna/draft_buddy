"""Durable JSON persistence helpers for draft state files."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional

ARCHIVE_RETENTION = 50
ARCHIVE_GLOB = "draft_state_*.json"


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Result of loading draft state JSON from disk.

    Parameters
    ----------
    payload : dict
        Parsed draft state dictionary.
    source_path : str
        Path of the file that supplied the payload.
    recovered_from_fallback : bool
        Whether a non-primary candidate was used.
    warning : str or None
        Human-readable recovery warning when fallback was used.
    """

    payload: dict
    source_path: str
    recovered_from_fallback: bool
    warning: Optional[str] = None


def write_json_atomic(path: str, payload: dict) -> None:
    """Write JSON atomically with flush and fsync.

    Parameters
    ----------
    path : str
        Destination file path.
    payload : dict
        JSON-serializable payload.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _is_readable_json(path: str) -> bool:
    """Return whether a path exists and parses as a JSON object."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return isinstance(payload, dict)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def save_draft_state(primary_path: str, payload: dict, prev_path: Optional[str] = None) -> None:
    """Persist draft state, keeping a rolling previous copy when possible.

    Parameters
    ----------
    primary_path : str
        Active draft state file.
    payload : dict
        Serialized draft state.
    prev_path : str, optional
        Rolling previous-file path. When set and the current primary is
        readable JSON, it is copied there before overwrite.
    """
    if prev_path and _is_readable_json(primary_path):
        prev_dir = os.path.dirname(prev_path) or "."
        os.makedirs(prev_dir, exist_ok=True)
        shutil.copy2(primary_path, prev_path)
    write_json_atomic(primary_path, payload)


def archive_draft_state(
    primary_path: str,
    saved_states_dir: str,
    retention: int = ARCHIVE_RETENTION,
) -> Optional[str]:
    """Copy the current primary draft state into a timestamped archive.

    Parameters
    ----------
    primary_path : str
        Active draft state file.
    saved_states_dir : str
        Directory for timestamped archives.
    retention : int, optional
        Maximum number of archives to retain.

    Returns
    -------
    str or None
        Archive path when a copy was written.
    """
    if not _is_readable_json(primary_path):
        return None
    os.makedirs(saved_states_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = os.path.join(saved_states_dir, f"draft_state_{stamp}.json")
    if os.path.exists(archive_path):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        archive_path = os.path.join(saved_states_dir, f"draft_state_{stamp}.json")
    shutil.copy2(primary_path, archive_path)
    prune_archives(saved_states_dir, retention=retention)
    return archive_path


def list_archive_paths(saved_states_dir: str) -> List[str]:
    """Return archive paths newest-first.

    Parameters
    ----------
    saved_states_dir : str
        Archive directory.

    Returns
    -------
    list of str
        Matching archive paths sorted by archive filename descending.
        Filenames embed ``YYYY-MM-DD_HH-MM-SS`` stamps, so name order is
        deterministic even when filesystem mtimes collide.
    """
    directory = Path(saved_states_dir)
    if not directory.is_dir():
        return []
    archives = [path for path in directory.glob(ARCHIVE_GLOB) if path.is_file()]
    archives.sort(key=lambda path: path.name, reverse=True)
    return [str(path) for path in archives]


def prune_archives(saved_states_dir: str, retention: int = ARCHIVE_RETENTION) -> None:
    """Delete oldest archives beyond the retention limit.

    Parameters
    ----------
    saved_states_dir : str
        Archive directory.
    retention : int, optional
        Maximum number of archives to keep.
    """
    archives = list_archive_paths(saved_states_dir)
    for stale_path in archives[retention:]:
        try:
            os.remove(stale_path)
        except OSError:
            pass


def iter_load_candidates(
    primary_path: str,
    prev_path: Optional[str] = None,
    saved_states_dir: Optional[str] = None,
) -> Iterable[tuple[str, bool]]:
    """Yield ``(path, is_primary)`` candidates in recovery order.

    Parameters
    ----------
    primary_path : str
        Active draft state file.
    prev_path : str, optional
        Rolling previous file.
    saved_states_dir : str, optional
        Timestamped archive directory.

    Yields
    ------
    tuple of (str, bool)
        Candidate path and whether it is the primary file.
    """
    yield primary_path, True
    if prev_path:
        yield prev_path, False
    if saved_states_dir:
        for archive_path in list_archive_paths(saved_states_dir):
            yield archive_path, False


def read_json_dict(path: str) -> dict:
    """Read a JSON object from disk.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    FileNotFoundError
        When the path does not exist.
    ValueError
        When the file is empty, unreadable, or not a JSON object.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    except OSError as error:
        raise ValueError(f"Unable to read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Draft state file must contain a JSON object: {path}")
    if not payload:
        raise ValueError(f"Draft state file is empty: {path}")
    return payload


def load_draft_state_payload(
    primary_path: str,
    prev_path: Optional[str] = None,
    saved_states_dir: Optional[str] = None,
    validate_payload: Optional[Callable[[dict], None]] = None,
) -> Optional[LoadResult]:
    """Load draft state JSON with primary → prev → archive recovery.

    Parameters
    ----------
    primary_path : str
        Active draft state file.
    prev_path : str, optional
        Rolling previous file.
    saved_states_dir : str, optional
        Timestamped archive directory.
    validate_payload : callable, optional
        Optional validator that raises ``ValueError`` on invalid payloads.

    Returns
    -------
    LoadResult or None
        Loaded payload when any candidate succeeds; ``None`` when no file exists.

    Raises
    ------
    ValueError
        When candidates exist but all fail to load or validate.
    """
    candidates = list(iter_load_candidates(primary_path, prev_path, saved_states_dir))
    existing = [(path, is_primary) for path, is_primary in candidates if os.path.isfile(path)]
    if not existing:
        return None

    errors: List[str] = []
    for path, is_primary in existing:
        try:
            payload = read_json_dict(path)
            if validate_payload is not None:
                validate_payload(payload)
            if is_primary:
                return LoadResult(
                    payload=payload,
                    source_path=path,
                    recovered_from_fallback=False,
                    warning=None,
                )
            warning = (
                f"Draft state primary unreadable or invalid; recovered from {os.path.basename(path)}."
            )
            return LoadResult(
                payload=payload,
                source_path=path,
                recovered_from_fallback=True,
                warning=warning,
            )
        except (ValueError, OSError, TypeError) as error:
            errors.append(f"{path}: {error}")
            continue

    detail = "; ".join(errors) if errors else "no readable candidates"
    raise ValueError(f"Could not load draft state from any candidate ({detail}).")
