"""Tests for the in-season trades CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.inseason.cli import _print_proposals, format_player_list, main, write_proposals_csv
from draft_buddy.inseason.store import InSeasonStore


def _store(config) -> InSeasonStore:
    """Build a store pointed at the test runtime paths."""
    return InSeasonStore(config.paths.INSEASON_STATE_FILE, config.paths.INSEASON_PROJECTIONS_FILE)


def _four_team_payload() -> dict:
    """Return a week-14 snapshot covering the four test teams."""
    return {
        "current_week": 14,
        "teams": [
            {"team_id": 1, "wins": 7, "losses": 6, "player_ids": [1, 2, 3, 4]},
            {"team_id": 2, "wins": 6, "losses": 7, "player_ids": [5, 6, 7, 8]},
            {"team_id": 3, "wins": 5, "losses": 8, "player_ids": [9, 10, 11, 12]},
            {"team_id": 4, "wins": 4, "losses": 9, "player_ids": [13, 14, 15, 16]},
        ],
    }


def test_search_text_prints_name_and_range() -> None:
    """Verify a proposal line names the player and prints the weekly range."""
    catalog = PlayerCatalog(
        [
            Player(
                player_id=1,
                name="QB One",
                position="QB",
                projected_points=10.0,
                weekly_points_median=12.0,
                weekly_points_mean=13.0,
                weekly_points_floor=8.0,
                weekly_points_ceiling=18.0,
                current_season_ppg=16.0,
                current_season_games=4,
            )
        ]
    )
    stdout = StringIO()
    _print_proposals(
        {
            "proposals": [
                {
                    "partner_team_id": 2,
                    "user_win_prob_delta": 0.02,
                    "partner_win_prob_delta": -0.01,
                    "user_sends_player_ids": [1],
                    "partner_sends_player_ids": [],
                }
            ]
        },
        stdout,
        catalog,
        {2: "Noah"},
    )

    assert "QB One" in stdout.getvalue() and "floor" in stdout.getvalue()


def test_write_proposals_csv_includes_player_name(tmp_path) -> None:
    """Verify a search CSV keeps the player name and the range columns."""
    catalog = PlayerCatalog(
        [
            Player(
                player_id=1,
                name="QB One",
                position="QB",
                projected_points=10.0,
                weekly_points_median=12.0,
                weekly_points_mean=13.0,
                weekly_points_floor=8.0,
                weekly_points_ceiling=18.0,
                current_season_ppg=16.0,
                current_season_games=4,
            )
        ]
    )
    destination = tmp_path / "trades.csv"
    write_proposals_csv(
        str(destination),
        [
            {
                "partner_team_id": 2,
                "user_win_prob_delta": 0.065,
                "partner_win_prob_delta": -0.03,
                "user_expected_wins_delta": 0.4,
                "partner_expected_wins_delta": -0.1,
                "user_starter_points_delta": 5.0,
                "partner_starter_points_delta": -2.0,
                "user_sends_player_ids": [1],
                "partner_sends_player_ids": [],
            }
        ],
        catalog,
        {2: "Noah"},
    )

    text = destination.read_text(encoding="utf-8")
    header = text.splitlines()[0].split(",")

    assert header[0] == "you_playoff_pp"
    assert header.index("send_1_name") < header.index("send_1_player_id")
    assert header.index("you_expected_wins_delta") < header.index("send_1_name")
    assert "QB One" in text


def test_format_player_list_uses_catalog_names(player_catalog) -> None:
    """Verify trade lines print position and name instead of raw ids."""
    assert format_player_list([1, 2], player_catalog) == "QB QB One, RB RB One"


def test_cli_snapshot_requires_sleeper_id(config, player_catalog) -> None:
    """Verify snapshot exits when the league has no Sleeper id."""
    stderr = StringIO()
    code = main(
        ["snapshot"],
        config=config,
        catalog=player_catalog,
        store=_store(config),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2


def test_cli_load_rejects_unmatched_player_names(config, player_catalog, tmp_path: Path) -> None:
    """Verify load fails closed on unmatched pasted names."""
    snapshot_path = tmp_path / "paste.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "current_week": 4,
                "teams": [{"team_id": 1, "player_names": ["Nobody McGhost"]}],
            }
        ),
        encoding="utf-8",
    )
    stderr = StringIO()
    code = main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=_store(config),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2


def test_cli_load_accepts_player_ids(config, player_catalog, tmp_path: Path) -> None:
    """Verify load persists an id-based snapshot."""
    snapshot_path = tmp_path / "paste.json"
    snapshot_path.write_text(json.dumps(_four_team_payload()), encoding="utf-8")
    code = main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=_store(config),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0


def test_cli_load_prints_file_schedule_source(config, player_catalog, tmp_path: Path) -> None:
    """Verify load names a pasted matchup as file-sourced."""
    snapshot_path = tmp_path / "paste.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "current_week": 4,
                "teams": [{"team_id": 1, "player_ids": [1]}],
                "matchups": [{"week": 4, "home": "Team 1", "away": "Team 2"}],
            }
        ),
        encoding="utf-8",
    )
    stdout = StringIO()
    main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=_store(config),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert "file" in stdout.getvalue()


def test_cli_show_fails_without_snapshot(config, player_catalog) -> None:
    """Verify show exits when no snapshot has been loaded."""
    code = main(
        ["show"],
        config=config,
        catalog=player_catalog,
        store=_store(config),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 2


def test_cli_evaluate_returns_verdict(config, player_catalog, tmp_path: Path) -> None:
    """Verify inbound eval prints a numeric verdict after load."""
    snapshot_path = tmp_path / "paste.json"
    snapshot_path.write_text(json.dumps(_four_team_payload()), encoding="utf-8")
    store = _store(config)
    main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    stdout = StringIO()
    main(
        ["evaluate", "--user", "Team 1", "--partner", "2", "--send", "TE One", "--receive", "6"],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=stdout,
        stderr=StringIO(),
    )
    first_line = stdout.getvalue().splitlines()[0]

    assert first_line.split(" · ")[0] in {"ACCEPT", "COUNTER", "REJECT"}


def test_cli_search_returns_proposals_key(config, player_catalog, tmp_path: Path) -> None:
    """Verify outbound search JSON includes a proposals list."""
    snapshot_path = tmp_path / "paste.json"
    payload = _four_team_payload()
    payload["teams"][0]["player_ids"] = [1]
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    store = _store(config)
    main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    stdout = StringIO()
    main(
        ["search", "--user", "1", "--json"],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=stdout,
        stderr=StringIO(),
    )

    assert "proposals" in json.loads(stdout.getvalue())


def test_cli_show_prints_week_and_source(config, player_catalog, tmp_path: Path) -> None:
    """Verify show prints a remaining week and where that slate came from."""
    snapshot_path = tmp_path / "paste.json"
    snapshot_path.write_text(json.dumps(_four_team_payload()), encoding="utf-8")
    store = _store(config)
    main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    stdout = StringIO()
    code = main(
        ["show"],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=stdout,
        stderr=StringIO(),
    )
    text = stdout.getvalue()

    assert code == 0
    assert "week 14:" in text
    assert "round robin" in text


def test_cli_dry_run_prints_simulated_deal_count(config, player_catalog, tmp_path: Path) -> None:
    """Verify dry-run stops after printing how many deals would be simulated."""
    snapshot_path = tmp_path / "paste.json"
    payload = _four_team_payload()
    payload["teams"][0]["player_ids"] = [1]
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    store = _store(config)
    main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    stdout = StringIO()
    code = main(
        ["search", "--user", "1", "--dry-run"],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=stdout,
        stderr=StringIO(),
    )

    assert code == 0
    assert "Deals that will be simulated:" in stdout.getvalue()


def test_cli_search_rejects_an_empty_send_pool(config, player_catalog, tmp_path: Path) -> None:
    """Verify keeping every player exits before season sims."""
    snapshot_path = tmp_path / "paste.json"
    snapshot_path.write_text(json.dumps(_four_team_payload()), encoding="utf-8")
    store = _store(config)
    main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    stderr = StringIO()
    code = main(
        ["search", "--user", "1", "--keep", "1", "2", "3", "4"],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert "No players left to send" in stderr.getvalue()


def test_cli_needs_ranks_a_complementary_team_first(config, player_catalog, tmp_path: Path) -> None:
    """Verify needs --user lists the team with matching bench surplus first."""
    snapshot_path = tmp_path / "paste.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "current_week": 14,
                "teams": [
                    {"team_id": 1, "wins": 7, "losses": 6, "player_ids": [1, 2]},
                    {"team_id": 2, "wins": 6, "losses": 7, "player_ids": [5, 7, 11, 15]},
                    {"team_id": 3, "wins": 5, "losses": 8, "player_ids": [9, 10]},
                    {"team_id": 4, "wins": 4, "losses": 9, "player_ids": [13, 14]},
                ],
            }
        ),
        encoding="utf-8",
    )
    store = _store(config)
    main(
        ["load", "--file", str(snapshot_path)],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    stdout = StringIO()
    code = main(
        ["needs", "--user", "1"],
        config=config,
        catalog=player_catalog,
        store=store,
        stdout=stdout,
        stderr=StringIO(),
    )
    complements = stdout.getvalue().split("Complements:", 1)[1]

    assert code == 0
    assert complements.strip().splitlines()[0].startswith("Team 2")
