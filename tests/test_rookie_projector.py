"""Tests for rookie projection strategies."""

from unittest.mock import Mock

import pandas as pd

from draft_buddy.data_pipeline.rookie_projector import RookieProjector


def test_estimate_by_draft_slot_assigns_pick_one_to_scale_max_percentile():
    """Verify top pick uses configured top percentile for projection."""
    projector = RookieProjector(scale_min=20, scale_max=80, udfa_percentile=75)
    veterans_df = pd.DataFrame(
        [
            {"position": "QB", "total_pts": 100.0},
            {"position": "QB", "total_pts": 200.0},
        ]
    )
    rookies_df = pd.DataFrame(
        [
            {"player_id": 1, "position": "QB", "draft_number": 1},
            {"player_id": 2, "position": "QB", "draft_number": 100},
        ]
    )
    projected = projector.estimate_by_draft_slot(rookies_df, veterans_df)
    projected_pick_one = float(projected[projected["player_id"] == 1]["total_pts"].iloc[0])

    assert projected_pick_one == 180.0


def test_project_rookies_with_adp_interpolates_midpoint_between_veterans():
    """Verify ADP interpolation returns midpoint between adjacent veterans."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame(
            [
                {"player_id": 10, "position": "QB", "AVG": 10.0, "is_rookie": False, "total_pts": 100.0},
                {"player_id": 20, "position": "QB", "AVG": 20.0, "is_rookie": True, "total_pts": None},
                {"player_id": 30, "position": "QB", "AVG": 30.0, "is_rookie": False, "total_pts": 200.0},
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame(
        [
            {"player_id": 10, "position": "QB", "total_pts": 100.0},
            {"player_id": 20, "position": "QB", "total_pts": None},
            {"player_id": 30, "position": "QB", "total_pts": 200.0},
        ]
    )
    projected = projector._project_rookies_with_adp(draft_players_df, "fake_adp.csv")
    rookie_points = float(projected[projected["player_id"] == 20]["total_pts"].iloc[0])

    assert rookie_points == 150.0


def test_project_rookies_hybrid_averages_draft_and_adp_predictions(monkeypatch):
    """Verify hybrid mode averages draft-based and ADP-based projections."""
    projector = RookieProjector(adp_matcher=Mock())
    draft_players_df = pd.DataFrame(
        [
            {"player_id": 1, "position": "QB", "is_rookie_original": False, "draft_number": None, "total_pts": 120.0},
            {"player_id": 2, "position": "QB", "is_rookie_original": True, "draft_number": 1, "total_pts": None},
        ]
    )

    def fake_adp_projection(*args, **kwargs):
        del args, kwargs
        return pd.DataFrame(
            [
                {"player_id": 1, "position": "QB", "is_rookie_original": False, "draft_number": None, "total_pts": 120.0},
                {"player_id": 2, "position": "QB", "is_rookie_original": True, "draft_number": 1, "total_pts": 200.0},
            ]
        )

    def fake_draft_projection(rookies_df, veterans_df):
        del veterans_df
        out = rookies_df.copy()
        out["total_pts"] = 100.0
        return out

    monkeypatch.setattr(projector, "_project_rookies_with_adp", fake_adp_projection)
    monkeypatch.setattr(projector, "estimate_by_draft_slot", fake_draft_projection)
    projected = projector.project_rookies(draft_players_df, method="hybrid", adp_filepath="fake_adp.csv")
    rookie_points = float(projected[projected["player_id"] == 2]["total_pts"].iloc[0])

    assert rookie_points == 150.0


def test_estimate_by_draft_slot_returns_zero_when_position_has_no_veterans():
    """Verify rookie points default to zero when no veteran baseline exists."""
    projector = RookieProjector()
    rookies_df = pd.DataFrame([{"player_id": 1, "position": "TE", "draft_number": 10}])
    veterans_df = pd.DataFrame([{"position": "QB", "total_pts": 200.0}])
    projected = projector.estimate_by_draft_slot(rookies_df, veterans_df)

    assert float(projected.iloc[0]["total_pts"]) == 0.0


def test_estimate_by_draft_slot_uses_udfa_percentile_for_missing_draft_number():
    """Verify missing draft numbers use configured UDFA percentile."""
    projector = RookieProjector(udfa_percentile=50)
    rookies_df = pd.DataFrame([{"player_id": 1, "position": "QB", "draft_number": None}])
    veterans_df = pd.DataFrame([{"position": "QB", "total_pts": 100.0}, {"position": "QB", "total_pts": 200.0}])
    projected = projector.estimate_by_draft_slot(rookies_df, veterans_df)

    assert float(projected.iloc[0]["draft_percentile"]) == 50.0


def test_project_rookies_with_adp_returns_input_when_no_file_or_matcher():
    """Verify ADP projection short-circuits without required inputs."""
    projector = RookieProjector(adp_matcher=None)
    draft_players_df = pd.DataFrame([{"player_id": 1, "total_pts": None}])
    projected = projector._project_rookies_with_adp(draft_players_df, "")

    assert projected.equals(draft_players_df)


def test_project_rookies_with_adp_returns_input_when_merged_matches_are_empty():
    """Verify empty matcher output preserves original player DataFrame."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame([{"player_id": 1, "total_pts": None}])
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert projected["total_pts"].isna().all()


def test_project_rookies_with_adp_returns_input_when_rank_column_missing():
    """Verify ADP projection exits when neither AVG nor Rank exists."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame([{"player_id": 1, "position": "QB", "is_rookie": True, "total_pts": None}]),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame([{"player_id": 1, "position": "QB", "total_pts": None}])
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert projected["total_pts"].isna().all()


def test_project_rookies_with_adp_uses_lower_veteran_when_upper_missing():
    """Verify interpolation falls back to nearest lower veteran."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame(
            [
                {"player_id": 10, "position": "QB", "AVG": 10.0, "is_rookie": False, "total_pts": 100.0},
                {"player_id": 20, "position": "QB", "AVG": 30.0, "is_rookie": True, "total_pts": None},
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame(
        [{"player_id": 10, "position": "QB", "total_pts": 100.0}, {"player_id": 20, "position": "QB", "total_pts": None}]
    )
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert float(projected[projected["player_id"] == 20]["total_pts"].iloc[0]) == 100.0


def test_project_rookies_with_adp_uses_upper_veteran_when_lower_missing():
    """Verify interpolation falls back to nearest upper veteran."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame(
            [
                {"player_id": 20, "position": "QB", "AVG": 10.0, "is_rookie": True, "total_pts": None},
                {"player_id": 30, "position": "QB", "AVG": 20.0, "is_rookie": False, "total_pts": 200.0},
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame(
        [{"player_id": 20, "position": "QB", "total_pts": None}, {"player_id": 30, "position": "QB", "total_pts": 200.0}]
    )
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert float(projected[projected["player_id"] == 20]["total_pts"].iloc[0]) == 200.0


def test_project_rookies_with_adp_uses_position_median_when_interpolation_invalid():
    """Verify invalid interpolation falls back to veteran median by position."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame(
            [
                {"player_id": 10, "position": "QB", "AVG": 10.0, "is_rookie": False, "total_pts": 100.0},
                {"player_id": 20, "position": "QB", "AVG": 10.0, "is_rookie": True, "total_pts": None},
                    {"player_id": 11, "position": "QB", "AVG": 10.0, "is_rookie": False, "total_pts": 200.0},
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame(
        [
            {"player_id": 10, "position": "QB", "total_pts": 100.0},
            {"player_id": 11, "position": "QB", "total_pts": 200.0},
            {"player_id": 20, "position": "QB", "total_pts": None},
        ]
    )
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert float(projected[projected["player_id"] == 20]["total_pts"].iloc[0]) == 150.0


def test_project_rookies_returns_input_when_no_rookies_exist():
    """Verify projection short-circuits when draft set has no rookies."""
    projector = RookieProjector()
    draft_players_df = pd.DataFrame([{"player_id": 1, "is_rookie_original": False, "total_pts": 100.0}])
    projected = projector.project_rookies(draft_players_df, method="draft")

    assert projected.equals(draft_players_df)


def test_project_rookies_draft_method_fills_rookies_from_draft_slot():
    """Verify draft method projects rookie totals."""
    projector = RookieProjector()
    draft_players_df = pd.DataFrame(
        [
            {"player_id": 1, "position": "QB", "is_rookie_original": False, "draft_number": None, "total_pts": 100.0},
            {"player_id": 2, "position": "QB", "is_rookie_original": True, "draft_number": 1, "total_pts": None},
        ]
    )
    projected = projector.project_rookies(draft_players_df, method="draft")

    assert projected["total_pts"].notna().all()


def test_project_rookies_adp_method_falls_back_to_draft_for_unfilled_rookies(monkeypatch):
    """Verify adp method uses draft-slot fallback when ADP leaves NaN rookies."""
    projector = RookieProjector(adp_matcher=Mock())
    draft_players_df = pd.DataFrame(
        [
            {"player_id": 1, "position": "QB", "is_rookie_original": False, "draft_number": None, "total_pts": 100.0},
            {"player_id": 2, "position": "QB", "is_rookie_original": True, "draft_number": 1, "total_pts": None},
        ]
    )
    monkeypatch.setattr(projector, "_project_rookies_with_adp", lambda *args, **kwargs: draft_players_df.copy())
    monkeypatch.setattr(
        projector,
        "estimate_by_draft_slot",
        lambda rookies_df, veterans_df: rookies_df.assign(total_pts=120.0),
    )
    projected = projector.project_rookies(draft_players_df, method="adp", adp_filepath="fake.csv")

    assert float(projected[projected["player_id"] == 2]["total_pts"].iloc[0]) == 120.0


def test_project_rookies_with_adp_returns_input_when_position_columns_missing():
    """Verify ADP projection exits when both Pos and position are absent."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame([{"player_id": 1, "AVG": 10.0, "is_rookie": True, "total_pts": None}]),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame([{"player_id": 1, "total_pts": None}])
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert projected["total_pts"].isna().all()


def test_project_rookies_with_adp_returns_input_when_no_veterans_exist_in_group():
    """Verify groups with no veteran anchors do not project rookie points."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame([{"player_id": 1, "position": "QB", "AVG": 10.0, "is_rookie": True, "total_pts": None}]),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame([{"player_id": 1, "position": "QB", "total_pts": None}])
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert projected["total_pts"].isna().all()


def test_project_rookies_with_adp_skips_rookie_rows_with_missing_adp_values():
    """Verify rookies with missing ADP remain unprojected."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame(
            [
                {"player_id": 10, "position": "QB", "AVG": 5.0, "is_rookie": False, "total_pts": 100.0},
                {"player_id": 20, "position": "QB", "AVG": None, "is_rookie": True, "total_pts": None},
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame(
        [{"player_id": 10, "position": "QB", "total_pts": 100.0}, {"player_id": 20, "position": "QB", "total_pts": None}]
    )
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert projected[projected["player_id"] == 20]["total_pts"].isna().all()


def test_project_rookies_with_adp_returns_input_when_no_predictions_are_created():
    """Verify function exits when no rookies receive predictions."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (
        pd.DataFrame(
            [{"player_id": 10, "position": "QB", "AVG": 5.0, "is_rookie": False, "total_pts": 100.0}]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    projector = RookieProjector(adp_matcher=matcher)
    draft_players_df = pd.DataFrame([{"player_id": 10, "position": "QB", "total_pts": 100.0}])
    projected = projector._project_rookies_with_adp(draft_players_df, "adp.csv")

    assert projected.equals(draft_players_df)


def test_project_rookies_hybrid_falls_back_to_draft_for_remaining_nan_rows(monkeypatch):
    """Verify hybrid method applies fallback for unresolved NaN totals."""
    projector = RookieProjector(adp_matcher=Mock())
    draft_players_df = pd.DataFrame(
        [
            {"player_id": 1, "position": "QB", "is_rookie_original": False, "draft_number": None, "total_pts": 120.0},
            {"player_id": 2, "position": "QB", "is_rookie_original": True, "draft_number": 1, "total_pts": None},
        ]
    )
    monkeypatch.setattr(projector, "_project_rookies_with_adp", lambda *args, **kwargs: draft_players_df.copy())
    monkeypatch.setattr(
        projector,
        "estimate_by_draft_slot",
        lambda rookies_df, veterans_df: rookies_df.assign(total_pts=130.0),
    )
    projected = projector.project_rookies(draft_players_df, method="hybrid", adp_filepath="fake.csv")

    assert float(projected[projected["player_id"] == 2]["total_pts"].iloc[0]) == 130.0
