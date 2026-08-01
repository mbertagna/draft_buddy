"""Tests for the generate_projections entry-point script's helper functions."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import generate_projections


def test_generated_output_dir_creates_league_scoped_directory(tmp_path: Path, monkeypatch) -> None:
    """Verify the output directory is created from the league-scoped player CSV path."""
    monkeypatch.chdir(tmp_path)
    output_path = "data/leagues/red_league_10/generated/2026/generated_player_data.csv"

    output_dir = generate_projections.generated_output_dir(output_path)

    assert os.path.samefile(
        output_dir,
        tmp_path / "data" / "leagues" / "red_league_10" / "generated" / "2026",
    )


def test_save_missing_nflverse_stats_report_writes_csv_when_non_empty(tmp_path: Path) -> None:
    """Verify a non-empty report is written to the year-scoped directory."""
    missing_df = pd.DataFrame([{"player_id": 1, "player_display_name": "Vet"}])

    generate_projections.save_missing_nflverse_stats_report(missing_df, str(tmp_path))

    report_path = tmp_path / 'sleeper_players_missing_nflverse_stats.csv'
    assert report_path.exists()


def test_save_missing_nflverse_stats_report_skips_write_when_empty(tmp_path: Path) -> None:
    """Verify no file is written when there is nothing to report."""
    generate_projections.save_missing_nflverse_stats_report(pd.DataFrame(), str(tmp_path))

    assert list(tmp_path.iterdir()) == []


class FakeSleeperGateway:
    """In-memory stand-in for SleeperHttpGateway used in tests."""

    def __init__(self, cache_dir: str) -> None:
        _ = cache_dir

    def fetch_all_players(self) -> pd.DataFrame:
        """Return a small Sleeper directory fixture."""
        return pd.DataFrame(
            [
                {
                    "sleeper_id": "1",
                    "full_name": "Rostered Kicker",
                    "position": "K",
                    "team": "KC",
                    "status": "Active",
                    "injury_status": None,
                    "depth_chart_position": None,
                    "years_exp": 3,
                },
            ]
        )

    def fetch_league_rosters(self, league_id: str) -> pd.DataFrame:
        """Return a roster containing a player excluded by the base filter."""
        _ = league_id
        return pd.DataFrame([{"roster_id": 1, "sleeper_id": "1"}])


def test_check_sleeper_roster_coverage_writes_csv_when_gaps_found(tmp_path: Path, monkeypatch) -> None:
    """Verify a rostered-but-excluded player is written to the year-scoped directory."""
    monkeypatch.setattr(generate_projections, "SleeperHttpGateway", FakeSleeperGateway)

    generate_projections.check_sleeper_roster_coverage(
        cache_dir=str(tmp_path), sleeper_league_id="league123", output_dir=str(tmp_path)
    )

    excluded_path = tmp_path / 'sleeper_players_excluded_by_filter.csv'
    assert excluded_path.exists()


class FakeSleeperGatewayFullyCovered(FakeSleeperGateway):
    """Gateway variant where the rostered player is not excluded by the filter."""

    def fetch_all_players(self) -> pd.DataFrame:
        """Return a directory where the rostered player is included in the skill catalog."""
        players_df = super().fetch_all_players()
        players_df["position"] = "WR"
        return players_df


def test_check_sleeper_roster_coverage_skips_write_when_fully_covered(tmp_path: Path, monkeypatch) -> None:
    """Verify no file is written when every rostered player is in the catalog."""
    monkeypatch.setattr(generate_projections, "SleeperHttpGateway", FakeSleeperGatewayFullyCovered)

    generate_projections.check_sleeper_roster_coverage(
        cache_dir=str(tmp_path), sleeper_league_id="league123", output_dir=str(tmp_path)
    )

    assert list(tmp_path.iterdir()) == []
