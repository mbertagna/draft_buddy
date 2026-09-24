"""Command-line entry for in-season snapshots, projections, and trades."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TextIO

from draft_buddy.config import load_runtime_config
from draft_buddy.core.entities import PlayerCatalog
from draft_buddy.data.player_loader import load_player_catalog
from draft_buddy.inseason.espn_adapter import resolve_player_names, resolve_team_id
from draft_buddy.inseason.exceptions import InSeasonError
from draft_buddy.inseason.schedule import schedule_summary
from draft_buddy.inseason.service import InSeasonService
from draft_buddy.inseason.store import InSeasonStore


def build_parser() -> argparse.ArgumentParser:
    """Build the trades CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with snapshot, load, schedule, projections, show, evaluate, and search.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print machine-readable JSON instead of text.",
    )
    parser = argparse.ArgumentParser(
        description="In-season trade intelligence for the active DRAFT_BUDDY_LEAGUE.",
    )
    subparsers = parser.add_subparsers(dest="command")

    snapshot = subparsers.add_parser(
        "snapshot",
        parents=[shared],
        help="Pull live Sleeper rosters when the league has a Sleeper id.",
    )
    snapshot.add_argument("--week", type=int, default=None, help="NFL week override.")

    load = subparsers.add_parser(
        "load",
        parents=[shared],
        help="Load a manual JSON snapshot from a file.",
    )
    load.add_argument("--file", required=True, help="Path to the snapshot JSON.")

    schedule = subparsers.add_parser(
        "schedule",
        parents=[shared],
        help="Merge an optional remaining-matchup file onto the stored snapshot.",
    )
    schedule.add_argument("--file", required=True, help="JSON file with a matchups list.")

    projections = subparsers.add_parser(
        "projections",
        parents=[shared],
        help="Rebuild rest-of-season weekly projection vectors.",
    )
    projections.add_argument("--week", type=int, default=None, help="First remaining NFL week.")

    subparsers.add_parser("show", parents=[shared], help="Print the stored snapshot.")

    evaluate = subparsers.add_parser(
        "evaluate",
        parents=[shared],
        help="Evaluate an inbound N-for-M offer.",
    )
    evaluate.add_argument("--user", required=True, help="Your team id or manager name.")
    evaluate.add_argument("--partner", required=True, help="Partner team id or manager name.")
    evaluate.add_argument(
        "--send",
        nargs="+",
        required=True,
        help="Players you send (ids or names).",
    )
    evaluate.add_argument(
        "--receive",
        nargs="+",
        required=True,
        help="Players you receive (ids or names).",
    )
    evaluate.add_argument("--writeup", action="store_true", help="Request an LLM writeup.")

    search = subparsers.add_parser(
        "search",
        parents=[shared],
        help="Find outbound 1v1 / 2v1 / 1v2 / 2v2 trades.",
    )
    search.add_argument("--user", required=True, help="Your team id or manager name.")
    search.add_argument(
        "--partner",
        default=None,
        help="Only this team id or manager name. Omit to search every other team.",
    )
    search.add_argument("--writeup", action="store_true", help="Request pitches for the top three.")
    search.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Write every ranked trade to this CSV path.",
    )
    search.add_argument(
        "--exhaustive",
        action="store_true",
        help="Score every 1v1, 2v1, 1v2, and 2v2 skill-player deal. Ignores the pool and candidate cap.",
    )
    search.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process count for season sims. 0 uses every CPU. 1 stays on one process.",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    config=None,
    catalog: Optional[PlayerCatalog] = None,
    store: Optional[InSeasonStore] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Run one trades CLI command.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector excluding the program name.
    config : Config, optional
        Injected runtime config. Loaded from disk when omitted.
    catalog : PlayerCatalog, optional
        Injected player catalog.
    store : InSeasonStore, optional
        Injected snapshot store.
    stdout : text IO, optional
        Output stream. Defaults to ``sys.stdout``.
    stderr : text IO, optional
        Error stream. Defaults to ``sys.stderr``.

    Returns
    -------
    int
        Process exit code.
    """
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help(out)
        return 0
    try:
        runtime = config or load_runtime_config()
        players = catalog or load_player_catalog(
            runtime.paths.PLAYER_DATA_CSV, runtime.draft.MOCK_ADP_CONFIG
        )
        snapshot_store = store or InSeasonStore(
            runtime.paths.INSEASON_STATE_FILE,
            runtime.paths.INSEASON_PROJECTIONS_FILE,
        )
        service = InSeasonService(runtime, players, snapshot_store)
        payload = _dispatch(args, runtime, players, service, stderr=err)
    except (InSeasonError, OSError, json.JSONDecodeError, ValueError) as error:
        print(str(error), file=err)
        return 2
    if args.command == "search" and getattr(args, "csv_path", None):
        count = write_proposals_csv(
            args.csv_path,
            payload.get("proposals") or [],
            players,
            runtime.draft.TEAM_MANAGER_MAPPING,
        )
        payload = dict(payload)
        payload["csv_path"] = str(args.csv_path)
        payload["csv_count"] = count
    _print_result(
        args.command,
        payload,
        as_json=args.as_json,
        stdout=out,
        catalog=players,
        team_names=runtime.draft.TEAM_MANAGER_MAPPING,
    )
    return 0


def _dispatch(
    args: argparse.Namespace,
    config,
    catalog: PlayerCatalog,
    service: InSeasonService,
    stderr: Optional[TextIO] = None,
):
    """Run the selected subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    config : Config
        Runtime configuration.
    catalog : PlayerCatalog
        Player catalog.
    service : InSeasonService
        In-season facade.
    stderr : text IO, optional
        Stream for search progress.

    Returns
    -------
    mapping
        Payload printed as text or JSON.
    """
    if args.command == "snapshot":
        if not str(getattr(config.draft, "SLEEPER_LEAGUE_ID", "") or ""):
            raise InSeasonError(
                "Sleeper league id is not configured. "
                "Use `load --file` for non-Sleeper leagues."
            )
        snapshot = service.refresh_sleeper_snapshot(args.week)
        return {"snapshot": snapshot.to_dict()}
    if args.command == "load":
        payload = _read_snapshot_file(args.file)
        snapshot = service.refresh_espn_snapshot(payload)
        return {"snapshot": snapshot.to_dict()}
    if args.command == "schedule":
        raw = _read_json_file(args.file)
        if isinstance(raw, list):
            raw = {"matchups": raw}
        if not isinstance(raw, dict):
            raise ValueError("Schedule file must contain a matchups list.")
        snapshot = service.apply_schedule(raw)
        return {"snapshot": snapshot.to_dict()}
    if args.command == "projections":
        weekly = service.refresh_projections(args.week)
        return {"player_count": len(weekly)}
    if args.command == "show":
        return service.state_payload()
    team_names = config.draft.TEAM_MANAGER_MAPPING
    user_id = resolve_team_id(args.user, team_names)
    if args.command == "evaluate":
        partner_id = resolve_team_id(args.partner, team_names)
        return service.evaluate_trade(
            user_id,
            partner_id,
            resolve_player_tokens(args.send, catalog),
            resolve_player_tokens(args.receive, catalog),
            include_writeup=args.writeup,
        )
    if args.command == "search":
        workers = _worker_count(args.workers)
        progress = _search_progress(stderr) if stderr is not None else None
        if workers > 1 and stderr is not None:
            print(f"Scoring trades on {workers} processes.", file=stderr)
        partner_id = resolve_team_id(args.partner, team_names) if args.partner else None
        return service.search_trades(
            user_id,
            include_writeup=args.writeup,
            exhaustive=args.exhaustive,
            progress=progress,
            workers=workers,
            partner_team_id=partner_id,
        )
    raise InSeasonError(f"Unknown command: {args.command}")


def resolve_player_tokens(tokens: Sequence[str], catalog: PlayerCatalog) -> list[int]:
    """Resolve CLI player ids or names into catalog ids.

    Parameters
    ----------
    tokens : sequence of str
        Player ids or display names.
    catalog : PlayerCatalog
        Lookup catalog.

    Returns
    -------
    list of int
        Resolved player ids.
    """
    ids: list[int] = []
    names: list[str] = []
    for token in tokens:
        stripped = str(token).strip()
        if stripped.isdigit():
            ids.append(int(stripped))
        else:
            names.append(stripped)
    if names:
        ids.extend(resolve_player_names(names, catalog))
    return ids


def _read_json_file(path: str) -> Any:
    """Load a JSON file.

    Parameters
    ----------
    path : str
        Filesystem path.

    Returns
    -------
    object
        Parsed JSON value.
    """
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_snapshot_file(path: str) -> Mapping[str, Any]:
    """Load a manual snapshot JSON file.

    Parameters
    ----------
    path : str
        Filesystem path to a JSON object.

    Returns
    -------
    mapping
        Snapshot payload.
    """
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError("Snapshot file must contain a JSON object.")
    return payload


def _print_result(
    command: str,
    payload: Mapping[str, Any],
    as_json: bool,
    stdout: TextIO,
    catalog: Optional[PlayerCatalog] = None,
    team_names: Optional[Mapping[int, str]] = None,
) -> None:
    """Write JSON or a short human-readable summary.

    Parameters
    ----------
    command : str
        Subcommand that produced ``payload``.
    payload : mapping
        Service result.
    as_json : bool
        When true, dump ``payload`` unchanged.
    stdout : text IO
        Output stream.
    catalog : PlayerCatalog, optional
        Used to print player names in text mode.
    team_names : mapping of int to str, optional
        Manager names keyed by team id.
    """
    if as_json:
        printable = payload
        if command == "search":
            printable = _search_payload_with_ranges(payload, catalog)
        json.dump(printable, stdout, indent=2, default=str)
        stdout.write("\n")
        return
    if command == "snapshot":
        snapshot = payload["snapshot"]
        print(
            f"Saved {snapshot.get('platform')} snapshot · week {snapshot.get('current_week')} "
            f"· {len(snapshot.get('teams') or {})} teams",
            file=stdout,
        )
        print(f"Schedule: {schedule_summary(snapshot.get('remaining_matchups') or [])}", file=stdout)
        return
    if command == "load":
        snapshot = payload["snapshot"]
        print(
            f"Loaded {snapshot.get('platform')} snapshot · week {snapshot.get('current_week')} "
            f"· {len(snapshot.get('teams') or {})} teams",
            file=stdout,
        )
        print(f"Schedule: {schedule_summary(snapshot.get('remaining_matchups') or [])}", file=stdout)
        return
    if command == "schedule":
        snapshot = payload["snapshot"]
        print(
            f"Updated schedule · week {snapshot.get('current_week')} · "
            f"{schedule_summary(snapshot.get('remaining_matchups') or [])}",
            file=stdout,
        )
        return
    if command == "projections":
        print(f"Wrote {payload['player_count']} ROS vectors.", file=stdout)
        return
    if command == "show":
        _print_state(payload, stdout)
        return
    if command == "evaluate":
        _print_evaluation(payload, stdout)
        return
    if command == "search":
        _print_proposals(payload, stdout, catalog, team_names or {})


def _print_state(state: Mapping[str, Any], stdout: TextIO) -> None:
    """Print week, records, and roster names."""
    print(
        f"{state.get('platform')} · week {state.get('current_week')}",
        file=stdout,
    )
    print(f"Schedule: {schedule_summary(state.get('remaining_matchups') or [])}", file=stdout)
    for team in state.get("teams") or []:
        record = f"{team.get('wins', 0)}-{team.get('losses', 0)}"
        if team.get("ties"):
            record = f"{record}-{team['ties']}"
        print(f"#{team.get('team_id')} {team.get('display_name')} {record}", file=stdout)
        for player in team.get("players") or []:
            print(
                f"  {player.get('position')} {player.get('name')} ({player.get('ros_points')})",
                file=stdout,
            )


def _print_evaluation(evaluation: Mapping[str, Any], stdout: TextIO) -> None:
    """Print inbound verdict and drop suggestion."""
    delta = float(evaluation.get("win_probability_delta") or 0.0)
    wins = float(evaluation.get("expected_wins_delta") or 0.0)
    print(
        f"{evaluation.get('verdict')} · playoff {_format_playoff_delta(delta)} · wins {wins:+.2f}",
        file=stdout,
    )
    drops = evaluation.get("optimal_bench_drops") or {}
    names = drops.get("dropped_player_names") or []
    if names:
        print("Drop: " + ", ".join(names), file=stdout)
    else:
        print("No drops needed.", file=stdout)
    breakdown = evaluation.get("defensive_breakdown") or ""
    if breakdown:
        print(breakdown, file=stdout)


def _print_proposals(
    payload: Mapping[str, Any],
    stdout: TextIO,
    catalog: Optional[PlayerCatalog],
    team_names: Mapping[int, str],
) -> None:
    """Print ranked outbound proposals.

    Parameters
    ----------
    payload : mapping
        Search result with a ``proposals`` list.
    stdout : text IO
        Output stream.
    catalog : PlayerCatalog or None
        Name lookup. Missing players stay as ids.
    team_names : mapping of int to str
        Manager names keyed by team id.
    """
    proposals = list(payload.get("proposals") or [])
    if not proposals:
        print("No trades reached simulation.", file=stdout)
        return
    visible = proposals
    if payload.get("exhaustive"):
        visible = proposals[:20]
        csv_path = payload.get("csv_path")
        extra = f" Full list: {csv_path}." if csv_path else " Pass --csv to save every trade."
        print(
            f"Exhaustive search scored {len(proposals)} trades, sorted by your playoff gain. "
            f"Showing {len(visible)}.{extra}",
            file=stdout,
        )
    elif payload.get("csv_path"):
        print(f"Wrote {payload.get('csv_count', len(proposals))} trades to {payload['csv_path']}.", file=stdout)
    for proposal in visible:
        user_delta = float(proposal.get("user_win_prob_delta") or 0.0)
        partner_delta = float(proposal.get("partner_win_prob_delta") or 0.0)
        partner_id = int(proposal.get("partner_team_id") or 0)
        partner_label = team_names.get(partner_id) or f"team {partner_id}"
        print(
            f"vs {partner_label} · you {_format_playoff_delta(user_delta)} "
            f"· them {_format_playoff_delta(partner_delta)}",
            file=stdout,
        )
        send_ids = proposal.get("user_sends_player_ids") or []
        receive_ids = proposal.get("partner_sends_player_ids") or []
        print(f"  Send {format_player_list(send_ids, catalog)} for {format_player_list(receive_ids, catalog)}", file=stdout)
        for player_id in list(send_ids) + list(receive_ids):
            line = format_player_range(player_id, catalog)
            if line:
                print(f"    {line}", file=stdout)


def _worker_count(requested: int) -> int:
    """Return a process count, using every CPU when ``requested`` is 0.

    Parameters
    ----------
    requested : int
        CLI ``--workers`` value.

    Returns
    -------
    int
        Process count of at least 1.
    """
    if requested > 0:
        return requested
    return os.cpu_count() or 1


def _search_progress(stderr: TextIO):
    """Return a callback that prints Monte Carlo progress."""

    def report(completed: int, total: int) -> None:
        print(f"Simulated {completed}/{total} trades", file=stderr)

    return report


def write_proposals_csv(
    path: str,
    proposals: Sequence[Mapping[str, Any]],
    catalog: Optional[PlayerCatalog],
    team_names: Mapping[int, str],
) -> int:
    """Write one CSV row per ranked trade.

    Parameters
    ----------
    path : str
        Destination CSV path. Parent directories are created when missing.
    proposals : sequence of mapping
        Search proposals in rank order.
    catalog : PlayerCatalog or None
        Name and range lookup.
    team_names : mapping of int to str
        Manager names keyed by team id.

    Returns
    -------
    int
        Number of rows written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    player_fields = ("player_id", "position", "name", "2026_ppg", "median", "mean", "floor", "ceiling")
    fieldnames = [
        "rank",
        "partner",
        "partner_team_id",
        "you_playoff_pp",
        "them_playoff_pp",
        "you_expected_wins_delta",
        "them_expected_wins_delta",
        "you_starter_ros_delta",
        "them_starter_ros_delta",
    ]
    for side in ("send", "receive"):
        for slot in (1, 2):
            fieldnames.extend(f"{side}_{slot}_{field}" for field in player_fields)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, proposal in enumerate(proposals, start=1):
            partner_id = int(proposal.get("partner_team_id") or 0)
            row = {
                "rank": rank,
                "partner": team_names.get(partner_id) or str(partner_id),
                "partner_team_id": partner_id,
                "you_playoff_pp": _playoff_pp(proposal.get("user_win_prob_delta")),
                "them_playoff_pp": _playoff_pp(proposal.get("partner_win_prob_delta")),
                "you_expected_wins_delta": _plain_number(proposal.get("user_expected_wins_delta")),
                "them_expected_wins_delta": _plain_number(proposal.get("partner_expected_wins_delta")),
                "you_starter_ros_delta": _plain_number(proposal.get("user_starter_points_delta")),
                "them_starter_ros_delta": _plain_number(proposal.get("partner_starter_points_delta")),
            }
            row.update(_player_slot_fields("send", proposal.get("user_sends_player_ids") or [], catalog))
            row.update(_player_slot_fields("receive", proposal.get("partner_sends_player_ids") or [], catalog))
            writer.writerow(row)
    return len(proposals)


def _player_slot_fields(
    side: str,
    player_ids: Sequence[int],
    catalog: Optional[PlayerCatalog],
) -> dict[str, str]:
    """Return one column per player field for up to two players on a side.

    Parameters
    ----------
    side : str
        ``send`` or ``receive``.
    player_ids : sequence of int
        Players on that side, in offer order.
    catalog : PlayerCatalog or None
        Name and range lookup.

    Returns
    -------
    dict
        Slot columns. Unused slots are blank.
    """
    fields = ("player_id", "position", "name", "2026_ppg", "median", "mean", "floor", "ceiling")
    columns: dict[str, str] = {}
    for slot in (1, 2):
        player = None
        raw_id = ""
        if slot <= len(player_ids):
            raw_id = str(int(player_ids[slot - 1]))
            if catalog is not None:
                player = catalog.get(int(raw_id))
        prefix = f"{side}_{slot}_"
        if player is None and not raw_id:
            for field in fields:
                columns[prefix + field] = ""
            continue
        if player is None:
            columns[prefix + "player_id"] = raw_id
            columns[prefix + "position"] = ""
            columns[prefix + "name"] = ""
            for field in ("2026_ppg", "median", "mean", "floor", "ceiling"):
                columns[prefix + field] = ""
            continue
        columns[prefix + "player_id"] = str(player.player_id)
        columns[prefix + "position"] = player.position
        columns[prefix + "name"] = player.name
        columns[prefix + "2026_ppg"] = _optional_number(player.current_season_ppg)
        columns[prefix + "median"] = _optional_number(player.weekly_points_median)
        columns[prefix + "mean"] = _optional_number(player.weekly_points_mean)
        columns[prefix + "floor"] = _optional_number(player.weekly_points_floor)
        columns[prefix + "ceiling"] = _optional_number(player.weekly_points_ceiling)
    return columns


def _playoff_pp(value: Any) -> str:
    """Format a playoff-probability delta as percentage points."""
    return f"{float(value or 0.0) * 100.0:.1f}"


def _plain_number(value: Any) -> str:
    """Format a number with one decimal and no forced sign."""
    return f"{float(value or 0.0):.1f}"


def _optional_number(value: Optional[float]) -> str:
    """Format a player rate, leaving the cell blank when it is missing."""
    if value is None:
        return ""
    return f"{float(value):.1f}"


def format_player_range(player_id: int, catalog: Optional[PlayerCatalog]) -> str:
    """Format one player's current rate and two-year range.

    Parameters
    ----------
    player_id : int
        Catalog player id.
    catalog : PlayerCatalog or None
        Lookup catalog.

    Returns
    -------
    str
        Empty when the player or the range is missing.
    """
    if catalog is None:
        return ""
    player = catalog.get(int(player_id))
    if player is None:
        return ""
    current = "n/a" if player.current_season_ppg is None else f"{player.current_season_ppg:.1f}"
    return (
        f"{player.position} {player.name}: 2026 {current} ppg · "
        f"median {_fmt_ppg(player.weekly_points_median)} · "
        f"mean {_fmt_ppg(player.weekly_points_mean)} · "
        f"floor {_fmt_ppg(player.weekly_points_floor)} · "
        f"ceiling {_fmt_ppg(player.weekly_points_ceiling)}"
    )


def _fmt_ppg(value: Optional[float]) -> str:
    """Format a points-per-game number or ``n/a``."""
    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


def _search_payload_with_ranges(payload: Mapping[str, Any], catalog: Optional[PlayerCatalog]) -> dict:
    """Copy a search payload and attach per-player range fields."""
    enriched = dict(payload)
    proposals = []
    for proposal in payload.get("proposals") or []:
        row = dict(proposal)
        player_ids = list(row.get("user_sends_player_ids") or []) + list(row.get("partner_sends_player_ids") or [])
        row["player_ranges"] = [
            {
                "player_id": int(player_id),
                "label": _player_label(player_id, catalog),
                "current_season_ppg": None if catalog is None or catalog.get(int(player_id)) is None else catalog.get(int(player_id)).current_season_ppg,
                "median": None if catalog is None or catalog.get(int(player_id)) is None else catalog.get(int(player_id)).weekly_points_median,
                "mean": None if catalog is None or catalog.get(int(player_id)) is None else catalog.get(int(player_id)).weekly_points_mean,
                "floor": None if catalog is None or catalog.get(int(player_id)) is None else catalog.get(int(player_id)).weekly_points_floor,
                "ceiling": None if catalog is None or catalog.get(int(player_id)) is None else catalog.get(int(player_id)).weekly_points_ceiling,
            }
            for player_id in player_ids
        ]
        proposals.append(row)
    enriched["proposals"] = proposals
    return enriched


def format_player_list(player_ids: Sequence[int], catalog: Optional[PlayerCatalog]) -> str:
    """Format player ids as position and name.

    Parameters
    ----------
    player_ids : sequence of int
        Catalog player ids.
    catalog : PlayerCatalog or None
        Lookup catalog. Unknown ids are left numeric.

    Returns
    -------
    str
        Comma-separated labels.
    """
    return ", ".join(_player_label(player_id, catalog) for player_id in player_ids)


def _player_label(player_id: int, catalog: Optional[PlayerCatalog]) -> str:
    """Return ``POS Name`` or the raw id when the catalog has no match."""
    if catalog is None:
        return str(player_id)
    player = catalog.get(int(player_id))
    if player is None:
        return str(player_id)
    return f"{player.position} {player.name}"


def _format_playoff_delta(value: float) -> str:
    """Format a playoff-probability delta as percentage points."""
    points = value * 100.0
    return f"{points:+.1f} pp"


if __name__ == "__main__":
    sys.exit(main())
