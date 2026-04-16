"""Tests for fantasy roster legality rules."""


def test_can_draft_manual_true_when_primary_starter_slot_open(
    core_draft_state, core_player_map, core_rules_engine
):
    """Verify manual drafting is allowed with open starter slots."""
    assert core_rules_engine.can_draft_manual(core_draft_state, 1, "RB", core_player_map) is True


def test_can_draft_manual_false_when_total_roster_size_reached(
    core_draft_state, core_player_map, core_total_roster_size, core_rules_engine
):
    """Verify manual drafting is denied when total roster cap is reached."""
    roster = core_draft_state.get_rosters()[1]
    roster["PLAYERS"] = [core_player_map[2]] * core_total_roster_size

    assert core_rules_engine.can_draft_manual(core_draft_state, 1, "RB", core_player_map) is False


def test_can_draft_manual_false_when_no_available_players_match_position(
    core_draft_state, core_player_map, core_rules_engine
):
    """Verify manual drafting is denied if no available player matches the position."""
    core_draft_state.replace_available_player_ids({1, 2, 3, 4, 9, 10})

    assert core_rules_engine.can_draft_manual(core_draft_state, 1, "WR", core_player_map) is False


def test_can_draft_simulated_false_when_qb_position_limit_is_exceeded(
    core_draft_state, core_player_map, core_rules_engine
):
    """Verify simulated drafting prevents exceeding starter-plus-bench limits."""
    roster = core_draft_state.get_rosters()[1]
    roster["QB"] = 2
    roster["PLAYERS"] = [core_player_map[1], core_player_map[1]]

    assert core_rules_engine.can_draft_simulated(core_draft_state, 1, "QB", core_player_map) is False


def test_can_draft_manual_true_when_flex_slot_available_for_rb(
    core_draft_state, core_player_map, core_rules_engine
):
    """Verify RB is still draftable when RB starters are full but FLEX is open."""
    roster = core_draft_state.get_rosters()[1]
    roster["RB"] = 2
    roster["FLEX"] = 0
    roster["PLAYERS"] = [core_player_map[2], core_player_map[3]]

    assert core_rules_engine.can_draft_manual(core_draft_state, 1, "RB", core_player_map) is True


def test_can_draft_manual_uses_bench_path_for_non_flex_position(core_draft_state, core_player_map, core_rules_engine):
    """Verify manual drafting can use bench when non-flex starters are full."""
    roster = core_draft_state.get_rosters()[1]
    roster["QB"] = 1
    roster["PLAYERS"] = [core_player_map[1], core_player_map[2], core_player_map[3]]

    assert core_rules_engine.can_draft_manual(core_draft_state, 1, "QB", core_player_map) is True


def test_can_draft_simulated_true_when_primary_full_but_flex_open(
    core_draft_state, core_player_map, core_rules_engine
):
    """Verify simulated drafting can route eligible position into FLEX."""
    roster = core_draft_state.get_rosters()[1]
    roster["WR"] = 2
    roster["FLEX"] = 0
    roster["PLAYERS"] = [core_player_map[5], core_player_map[6]]

    assert core_rules_engine.can_draft_simulated(core_draft_state, 1, "WR", core_player_map) is True


def test_can_draft_simulated_false_when_total_roster_size_reached(
    core_draft_state, core_player_map, core_total_roster_size, core_rules_engine
):
    """Verify simulated drafting is denied when roster cap is reached."""
    roster = core_draft_state.get_rosters()[1]
    roster["PLAYERS"] = [core_player_map[2]] * core_total_roster_size

    assert core_rules_engine.can_draft_simulated(core_draft_state, 1, "RB", core_player_map) is False


def test_can_draft_simulated_true_when_primary_starter_slot_open(
    core_draft_state, core_player_map, core_rules_engine
):
    """Verify simulated drafting is allowed when primary starter slot is open."""
    assert core_rules_engine.can_draft_simulated(core_draft_state, 1, "TE", core_player_map) is True


def test_can_draft_simulated_false_when_no_available_players_match_position(
    core_draft_state, core_player_map, core_rules_engine
):
    """Verify simulated drafting is denied if no matching position is available."""
    core_draft_state.replace_available_player_ids({1, 2, 3, 4, 9, 10})

    assert core_rules_engine.can_draft_simulated(core_draft_state, 1, "WR", core_player_map) is False

