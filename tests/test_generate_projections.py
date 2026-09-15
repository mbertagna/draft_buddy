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


def test_parse_cli_args_defaults_year_from_season() -> None:
    """Verify the projections CLI year defaults to the active season overlay."""
    args = generate_projections.parse_cli_args([], default_year=2026)

    assert args.year == 2026
    assert args.sleeper_league_id is None


def test_fantasypros_scoring_code_maps_reception_scoring() -> None:
    """Verify league reception scoring maps to FantasyPros scoring codes."""
    from draft_buddy.config import load_runtime_config

    ppr = load_runtime_config(league_id="red_league_10", season=2026)
    half = load_runtime_config(league_id="redraft_nbfl_12", season=2026)

    assert generate_projections.fantasypros_scoring_code(ppr) == "PPR"
    assert generate_projections.fantasypros_scoring_code(half) == "HALF"


def test_fantasypros_rankings_to_adp_dataframe_shapes_matcher_input() -> None:
    """Verify normalized FantasyPros API rows convert to ADP matcher columns."""
    rankings_df = pd.DataFrame(
        [
            {
                "player_name": "Player A",
                "team": "BUF",
                "position": "RB",
                "adp": 12.5,
                "rank_std": 2.0,
                "rank_min": 10.0,
                "rank_max": 20.0,
                "source": "fantasypros_adp",
                "fetched_at": "2026-08-01T00:00:00+00:00",
            }
        ]
    )

    adp_df = generate_projections.fantasypros_rankings_to_adp_dataframe(rankings_df)

    assert adp_df.iloc[0]["Player"] == "Player A"
    assert adp_df.iloc[0]["AVG"] == 12.5
    assert adp_df.iloc[0]["market_source"] == "fantasypros_adp"


def test_resolve_adp_input_uses_fantasypros_api_when_available(monkeypatch, config) -> None:
    """Verify projection generation prefers API ADP rows when available."""

    class FakeFantasyProsGateway:
        """Gateway returning deterministic rankings."""

        def __init__(self, cache_dir: str) -> None:
            self.cache_dir = cache_dir

        def fetch_consensus_rankings(self, request, allow_stale_cache: bool = True):
            """Return normalized ranking rows."""
            _ = (request, allow_stale_cache)
            return pd.DataFrame(
                [
                    {
                        "player_name": "Player A",
                        "team": "BUF",
                        "position": "RB",
                        "adp": 12.5,
                        "source": "fantasypros_adp",
                        "fetched_at": "2026-08-01T00:00:00+00:00",
                    }
                ]
            )

    monkeypatch.setattr(generate_projections, "FantasyProsGateway", FakeFantasyProsGateway)

    adp_input = generate_projections.resolve_adp_input(config, 2026, "fallback.html")

    assert isinstance(adp_input, pd.DataFrame)
    assert adp_input.iloc[0]["AVG"] == 12.5


def test_resolve_adp_input_falls_back_to_html_when_api_unavailable(monkeypatch, config) -> None:
    """Verify projection generation keeps the HTML fallback on API failure."""

    class FakeFantasyProsGateway:
        """Gateway raising a market-data failure."""

        def __init__(self, cache_dir: str) -> None:
            self.cache_dir = cache_dir

        def fetch_consensus_rankings(self, request, allow_stale_cache: bool = True):
            """Raise an unavailable error."""
            _ = (request, allow_stale_cache)
            raise generate_projections.MarketDataUnavailable("no data")

    monkeypatch.setattr(generate_projections, "FantasyProsGateway", FakeFantasyProsGateway)

    assert generate_projections.resolve_adp_input(config, 2026, "fallback.html") == "fallback.html"


def test_resolve_sleeper_league_id_uses_overlay_when_cli_omitted() -> None:
    """Verify Sleeper roster coverage uses the season overlay league id."""
    from draft_buddy.config import load_runtime_config

    config = load_runtime_config(league_id="redraft_nbfl_12", season=2026)
    resolved = generate_projections.resolve_sleeper_league_id(None, config)

    assert resolved == config.draft.SLEEPER_LEAGUE_ID
    assert resolved


def test_resolve_sleeper_league_id_prefers_cli_over_overlay() -> None:
    """Verify an explicit CLI Sleeper league id wins over the overlay."""
    from draft_buddy.config import load_runtime_config

    config = load_runtime_config(league_id="redraft_nbfl_12", season=2026)
    resolved = generate_projections.resolve_sleeper_league_id("cli-league", config)

    assert resolved == "cli-league"


def test_resolve_sleeper_league_id_skips_empty_overlay() -> None:
    """Verify leagues without Sleeper sync do not enable roster coverage."""
    from draft_buddy.config import load_runtime_config

    config = load_runtime_config(league_id="red_league_10", season=2026)
    resolved = generate_projections.resolve_sleeper_league_id(None, config)

    assert resolved is None
