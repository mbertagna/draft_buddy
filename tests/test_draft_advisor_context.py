"""Tests for draft assistant context builder."""

from __future__ import annotations

from draft_buddy.core.entities import Player
from draft_buddy.data.insights.schemas import (
    Confidence,
    DepthRole,
    PlayerInsight,
    PlayingTimeTier,
    RecoveryStatus,
    RiskLevel,
)
from draft_buddy.web.draft_advisor_context import (
    DEFAULT_TOP_K,
    HIGH_PRIORITY_TOP_K,
    build_advisor_context,
    build_candidate_rows,
    collect_candidate_player_ids,
    format_league_format_blurb,
    position_top_k_map,
)


def _player(
    player_id: int,
    position: str,
    projected: float,
    adp: float,
    *,
    team: str | None = None,
    sleeper_status: str | None = None,
    sleeper_injury_status: str | None = None,
    sleeper_depth_chart_position: str | None = None,
) -> Player:
    """Build a test player."""
    return Player(
        player_id=player_id,
        name=f"Player {player_id}",
        position=position,
        projected_points=projected,
        adp=adp,
        team=team,
        sleeper_status=sleeper_status,
        sleeper_injury_status=sleeper_injury_status,
        sleeper_depth_chart_position=sleeper_depth_chart_position,
    )


def test_position_top_k_map_assigns_seven_to_top_two_positions() -> None:
    """Verify the two highest RL positions receive K=7."""
    top_k = position_top_k_map({"QB": 0.1, "RB": 0.5, "WR": 0.3, "TE": 0.1})

    assert top_k["RB"] == HIGH_PRIORITY_TOP_K
    assert top_k["WR"] == HIGH_PRIORITY_TOP_K
    assert top_k["QB"] == DEFAULT_TOP_K
    assert top_k["TE"] == DEFAULT_TOP_K


def test_position_top_k_map_tie_breaks_with_position_order() -> None:
    """Verify equal probabilities tie-break using QB, RB, WR, TE order."""
    top_k = position_top_k_map({"QB": 0.25, "RB": 0.25, "WR": 0.25, "TE": 0.25})

    assert top_k["QB"] == HIGH_PRIORITY_TOP_K
    assert top_k["RB"] == HIGH_PRIORITY_TOP_K


def test_format_league_format_blurb_labels_full_ppr() -> None:
    """Verify full PPR scoring produces a compact league format string."""
    blurb = format_league_format_blurb(
        scoring_rules={"receptions": 1.0},
        num_teams=10,
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 3},
        total_bench_size=7,
    )

    assert blurb == "Full PPR · 10 teams · starters QB1 RB2 WR2 TE1 FLEX3 · bench 7"


def test_format_league_format_blurb_labels_half_ppr() -> None:
    """Verify half PPR scoring is labeled correctly."""
    blurb = format_league_format_blurb(
        scoring_rules={"receptions": 0.5},
        num_teams=12,
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        total_bench_size=6,
    )

    assert "Half PPR" in blurb
    assert "12 teams" in blurb


def test_build_advisor_context_includes_glossary_and_roster(config) -> None:
    """Verify markdown context contains key sections."""
    players = [
        _player(1, "RB", 240.0, 2.0, team="KC", sleeper_status="Active"),
        _player(2, "WR", 230.0, 3.0, team="BUF", sleeper_injury_status="Questionable"),
    ]
    rows = build_candidate_rows(players, {"RB": 180.0, "WR": 170.0, "QB": 200.0, "TE": 120.0})
    top_k = position_top_k_map({"QB": 0.1, "RB": 0.6, "WR": 0.2, "TE": 0.1})
    ui_state = {
        "current_pick_number": 5,
        "current_pick_index": 4,
        "draft_order": [1, 2, 3, 4, 1, 2, 3, 4],
        "num_teams": 4,
        "snake_team_on_turn": 1,
        "override_active": False,
        "total_roster_size_per_team": 16,
        "team_rosters": {
            1: {"starters": {"RB": [players[0].to_dict()]}, "bench": []},
            2: {"starters": {}, "bench": []},
        },
        "roster_counts": {
            1: {"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
            2: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        },
        "team_bye_weeks": {1: {}, 2: {}},
    }
    insight = PlayerInsight(
        outlook_phrase="High-volume WR1",
        summary="Expected to lead the team in targets this season.",
        depth_role=DepthRole.STARTER,
        playing_time_tier=PlayingTimeTier.HIGH,
        injury_risk=RiskLevel.LOW,
        upside=RiskLevel.HIGH,
        recovery_status=RecoveryStatus.NA,
        overall_confidence=Confidence.HIGH,
    )
    context = build_advisor_context(
        ui_state=ui_state,
        advising_team_id=1,
        agent_team_id=1,
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        total_bench_size=config.draft.TOTAL_BENCH_SIZE,
        team_manager_mapping=config.draft.TEAM_MANAGER_MAPPING,
        candidate_rows=rows,
        baselines={"RB": 180.0, "WR": 170.0, "QB": 200.0, "TE": 120.0},
        top_k_by_position=top_k,
        rl_probs={"QB": 0.1, "RB": 0.6, "WR": 0.2, "TE": 0.1},
        rl_degraded=False,
        insights={2: insight},
        recent_picks=[
            {
                "pick_number": 4,
                "team_id": 2,
                "player_name": "Recent RB",
                "position": "RB",
            }
        ],
        league_format_blurb="Full PPR · 10 teams · starters QB1 RB2 WR2 TE1 FLEX3 · bench 7",
    )

    assert "## Field glossary" in context
    assert "Low gp_frac may reflect a past injury" in context
    assert "When insight fields are blank for a player, use stats only" in context
    assert "## League format" in context
    assert "Full PPR · 10 teams" in context
    assert "Picks until advising team's next selection" in context
    assert "## Advising team roster" in context
    assert "## Roster targets" in context
    assert "starters_req" in context
    assert "## Recent picks" in context
    assert "Recent RB" in context
    assert "## League snapshot (other teams)" in context
    assert "## Best available overall" in context
    assert "## Best for open starter needs" in context
    assert "RB — by VORP" in context
    assert "candidate shortlist K=7" in context
    assert "| nfl | status | injury | depth |" in context or "nfl | status | injury | depth" in context
    assert "KC" in context
    assert "Questionable" in context
    assert "Expected to lead the team in targets this season." in context
    assert "draft_lean" in context
    assert "handcuff" in context
    assert "evidence_as_of" in context
    assert "quick_take" in context
    assert "reasoning first" in context
    assert "Stage A — Eligibility" in context
    assert "Tier cliff only when ALL hold" in context
    assert "knows little" not in context
    assert '"by ADP" tables are market order' in context
    assert "room_at_pos = 0 is a soft cap" in context
    assert "late-draft trade hoarding" in context


def test_build_advisor_context_keeps_soft_capped_positions(config) -> None:
    """Verify room_at_pos=0 positions stay on the board for manual over-cap picks."""
    qb = _player(1, "QB", 300.0, 10.0)
    rb = _player(2, "RB", 240.0, 2.0)
    rows = build_candidate_rows([qb, rb], {"QB": 200.0, "RB": 180.0, "WR": 170.0, "TE": 120.0})
    roster_structure = {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 0}
    bench_maxes = {"QB": 0, "RB": 1, "WR": 1, "TE": 0}
    roster = {
        "starters": {"QB": [qb.to_dict()]},
        "bench": [],
        "players_flat": [qb.to_dict()],
    }
    ui_state = {
        "current_pick_number": 2,
        "current_pick_index": 1,
        "draft_order": [1, 2, 2, 1],
        "num_teams": 2,
        "snake_team_on_turn": 2,
        "override_active": False,
        "total_roster_size_per_team": 5,
        "team_rosters": {1: roster, 2: {"starters": {}, "bench": []}},
        "roster_counts": {
            1: {"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
            2: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        },
        "team_bye_weeks": {1: {}, 2: {}},
    }
    context = build_advisor_context(
        ui_state=ui_state,
        advising_team_id=1,
        agent_team_id=1,
        roster_structure=roster_structure,
        bench_maxes=bench_maxes,
        total_bench_size=2,
        team_manager_mapping={1: "Team 1", 2: "Team 2"},
        candidate_rows=rows,
        baselines={"QB": 200.0, "RB": 180.0, "WR": 170.0, "TE": 120.0},
        top_k_by_position={"QB": 5, "RB": 5, "WR": 5, "TE": 5},
        rl_probs={"QB": 0.25, "RB": 0.25, "WR": 0.25, "TE": 0.25},
        rl_degraded=False,
        insights={},
    )

    assert "position full — not draftable" not in context
    assert "QB — by VORP" in context
    assert "RB — by VORP" in context
    assert "manual drafting may pick them" in context
    assert "trade hoarding" in context


def test_render_candidate_table_omits_insight_columns_when_empty() -> None:
    """Verify insight-less tables drop the long insight column set."""
    from draft_buddy.web.draft_advisor_context import _render_candidate_table

    rows = build_candidate_rows(
        [_player(1, "RB", 240.0, 2.0, sleeper_status="Active")],
        {"RB": 180.0},
    )
    table = _render_candidate_table("RB — by VORP", rows, insights={})

    assert "outlook" not in table
    assert "summary" not in table
    assert "| vorp | adp | gp_frac | proj | bye |" in table


def test_picks_until_next_turn_counts_from_current_index() -> None:
    """Verify next-selection distance excludes the current pick index."""
    from draft_buddy.web.draft_advisor_context import _picks_until_next_turn

    assert _picks_until_next_turn([1, 2, 3, 4, 4, 3, 2, 1], 0, 1) == 7
    assert _picks_until_next_turn([1, 2, 3, 4, 4, 3, 2, 1], 1, 2) == 5
    assert _picks_until_next_turn([1, 2, 3, 4], 3, 4) is None


def test_format_position_targets_table_shows_remaining_room(config) -> None:
    """Verify roster target table includes per-position caps and open starter slots."""
    from draft_buddy.web.draft_advisor_context import _format_position_targets_table

    player = _player(1, "RB", 240.0, 2.0)
    roster = {"starters": {"RB": [player.to_dict()]}, "bench": [], "players_flat": [player.to_dict()]}
    table = _format_position_targets_table(
        roster=roster,
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 3},
        bench_maxes={"QB": 3, "RB": 8, "WR": 8, "TE": 4},
        total_bench_size=7,
        total_roster_size=16,
    )

    assert "| RB | 1 | 2 | 1 |" in table
    assert "room_at_pos" in table
    assert "does not block manual picks" in table


def test_collect_candidate_player_ids_unions_vorp_adp_and_overall() -> None:
    """Verify valid recommendation ids include ranking views and overall board."""
    players = [
        _player(1, "RB", 240.0, 2.0),
        _player(2, "RB", 220.0, 6.0),
        _player(3, "RB", 210.0, 8.0),
    ]
    rows = build_candidate_rows(players, {"RB": 180.0, "WR": 170.0, "QB": 200.0, "TE": 120.0})
    top_k = {"QB": 5, "RB": 2, "WR": 5, "TE": 5}
    valid_ids = collect_candidate_player_ids(rows, top_k)

    assert valid_ids == {1, 2, 3}


def test_collect_candidate_player_ids_includes_soft_capped_positions() -> None:
    """Verify players at room_at_pos=0 remain valid for manual over-cap recommendations."""
    players = [
        _player(1, "QB", 300.0, 10.0),
        _player(2, "RB", 240.0, 2.0),
    ]
    rows = build_candidate_rows(players, {"QB": 200.0, "RB": 180.0, "WR": 170.0, "TE": 120.0})
    qb = players[0]
    roster = {
        "starters": {"QB": [qb.to_dict()]},
        "bench": [],
        "players_flat": [qb.to_dict()],
    }
    valid_ids = collect_candidate_player_ids(
        rows,
        {"QB": 5, "RB": 5, "WR": 5, "TE": 5},
        roster=roster,
        roster_structure={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 0},
        bench_maxes={"QB": 0, "RB": 1, "WR": 1, "TE": 0},
    )

    assert valid_ids == {1, 2}


def test_collect_candidate_player_ids_includes_need_fill_players() -> None:
    """Verify open starter needs add eligible shortlist players."""
    players = [
        _player(10, "QB", 300.0, 20.0),
        _player(11, "QB", 280.0, 30.0),
        _player(12, "QB", 260.0, 40.0),
        _player(13, "QB", 240.0, 50.0),
        _player(20, "RB", 200.0, 5.0),
    ]
    rows = build_candidate_rows(
        players,
        {"RB": 150.0, "WR": 140.0, "QB": 220.0, "TE": 100.0},
    )
    top_k = {"QB": 1, "RB": 1, "WR": 1, "TE": 1}
    roster = {"starters": {}, "bench": []}
    roster_structure = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 0}

    valid_ids = collect_candidate_player_ids(
        rows,
        top_k,
        roster=roster,
        roster_structure=roster_structure,
        bench_maxes={"QB": 3, "RB": 8, "WR": 8, "TE": 4},
    )

    assert 10 in valid_ids
    assert 11 in valid_ids
    assert 12 in valid_ids
    assert 20 in valid_ids
