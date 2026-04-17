"""Tests for rookie projection behavior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from draft_buddy.data.rookie_projector import RookieProjector


def test_estimate_by_draft_slot_returns_zero_when_no_veterans() -> None:
    """Verify positions with no veteran history get zero projected points."""
    rookies_df = pd.DataFrame([{"position": "QB", "draft_number": 1}])
    veterans_df = pd.DataFrame(columns=["position", "total_pts"])
    result = RookieProjector().estimate_by_draft_slot(rookies_df, veterans_df)

    assert float(result.iloc[0]["total_pts"]) == 0.0


def test_estimate_by_draft_slot_scales_udfa_and_single_draft_slot() -> None:
    """Verify drafted and undrafted rookies use the expected percentile branches."""
    rookies_df = pd.DataFrame(
        [
            {"player_id": 1, "position": "QB", "draft_number": 10},
            {"player_id": 2, "position": "QB", "draft_number": None},
        ]
    )
    veterans_df = pd.DataFrame(
        [
            {"position": "QB", "total_pts": 100.0},
            {"position": "QB", "total_pts": 200.0},
            {"position": "QB", "total_pts": 300.0},
        ]
    )
    projector = RookieProjector(scale_min=5, udfa_percentile=75)

    result = projector.estimate_by_draft_slot(rookies_df, veterans_df)

    drafted_points = float(result.loc[result["player_id"] == 1, "total_pts"].iloc[0])
    udfa_points = float(result.loc[result["player_id"] == 2, "total_pts"].iloc[0])
    assert drafted_points == np.percentile(veterans_df["total_pts"], 95) and udfa_points == np.percentile(
        veterans_df["total_pts"], 75
    )


def test_project_rookies_with_adp_returns_input_when_matcher_or_path_missing() -> None:
    """Verify ADP projection exits early when the matcher or file path is unavailable."""
    dataframe = pd.DataFrame(
        [{"player_id": 1, "total_pts": None, "position": "QB", "is_rookie_original": True}]
    )

    assert RookieProjector()._project_rookies_with_adp(dataframe, "adp.csv").equals(dataframe)
    assert RookieProjector(adp_matcher=object())._project_rookies_with_adp(dataframe, "").equals(dataframe)


def test_project_rookies_with_adp_returns_input_on_empty_merge(monkeypatch) -> None:
    """Verify empty ADP merges leave the draft players unchanged."""
    projector = RookieProjector(
        adp_matcher=SimpleNamespace(
            merge_adp_data=lambda **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        )
    )
    dataframe = pd.DataFrame(
        [{"player_id": 1, "total_pts": None, "position": "QB", "is_rookie_original": True}]
    )

    result = projector._project_rookies_with_adp(dataframe, "adp.csv")

    assert result.equals(dataframe)


def test_project_rookies_with_adp_returns_input_when_required_columns_are_missing(monkeypatch) -> None:
    """Verify ADP projection exits when merged data lacks ADP or position columns."""
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "total_pts": 12.0, "position": "QB"},
            {"player_id": 2, "total_pts": None, "position": "QB"},
        ]
    )
    projector = RookieProjector(
        adp_matcher=SimpleNamespace(
            merge_adp_data=lambda **_kwargs: (
                pd.DataFrame([{"player_id": 1, "position": "QB", "total_pts": 12.0, "is_rookie": False}]),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        )
    )

    result = projector._project_rookies_with_adp(dataframe, "adp.csv")

    assert result.equals(dataframe)


def test_project_rookies_with_adp_interpolates_between_veteran_anchors() -> None:
    """Verify rookies between two veteran ADP anchors are linearly interpolated."""
    merged_df = pd.DataFrame(
        [
            {"player_id": 10, "Pos": "QB", "AVG": 10.0, "total_pts": 30.0, "is_rookie": False},
            {"player_id": 20, "Pos": "QB", "AVG": 15.0, "total_pts": None, "is_rookie": True},
            {"player_id": 30, "Pos": "QB", "AVG": 20.0, "total_pts": 10.0, "is_rookie": False},
        ]
    )
    projector = RookieProjector(
        adp_matcher=SimpleNamespace(
            merge_adp_data=lambda **_kwargs: (merged_df.copy(), pd.DataFrame(), pd.DataFrame())
        )
    )
    dataframe = pd.DataFrame(
        [
            {"player_id": 10, "total_pts": 30.0, "position": "QB"},
            {"player_id": 20, "total_pts": None, "position": "QB"},
            {"player_id": 30, "total_pts": 10.0, "position": "QB"},
        ]
    )

    result = projector._project_rookies_with_adp(dataframe, "adp.csv")

    assert float(result.loc[result["player_id"] == 20, "total_pts"].iloc[0]) == 20.0


def test_project_rookies_with_adp_uses_neighbor_or_position_median_fallbacks() -> None:
    """Verify ADP projection uses one-sided and median fallbacks when needed."""
    merged_df = pd.DataFrame(
        [
            {"player_id": 1, "Pos": "RB", "AVG": 5.0, "total_pts": 25.0, "is_rookie": False},
            {"player_id": 2, "Pos": "RB", "AVG": 7.0, "total_pts": None, "is_rookie": True},
            {"player_id": 3, "Pos": "WR", "AVG": 20.0, "total_pts": 9.0, "is_rookie": False},
            {"player_id": 4, "Pos": "WR", "AVG": 30.0, "total_pts": 15.0, "is_rookie": False},
            {"player_id": 5, "Pos": "WR", "AVG": None, "total_pts": None, "is_rookie": True},
        ]
    )
    projector = RookieProjector(
        adp_matcher=SimpleNamespace(
            merge_adp_data=lambda **_kwargs: (merged_df.copy(), pd.DataFrame(), pd.DataFrame())
        )
    )
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "total_pts": 25.0, "position": "RB"},
            {"player_id": 2, "total_pts": None, "position": "RB"},
            {"player_id": 3, "total_pts": 9.0, "position": "WR"},
            {"player_id": 4, "total_pts": 15.0, "position": "WR"},
            {"player_id": 5, "total_pts": None, "position": "WR"},
        ]
    )

    result = projector._project_rookies_with_adp(dataframe, "adp.csv")

    assert float(result.loc[result["player_id"] == 2, "total_pts"].iloc[0]) == 25.0 and pd.isna(
        result.loc[result["player_id"] == 5, "total_pts"].iloc[0]
    )


def test_project_rookies_adp_mode_falls_back_to_draft_slot_for_remaining_nan(monkeypatch) -> None:
    """Verify ADP mode fills any remaining rookie gaps with draft-slot estimates."""
    projector = RookieProjector(adp_matcher=object())
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "is_rookie_original": False, "total_pts": 20.0, "position": "QB", "draft_number": 1},
            {"player_id": 2, "is_rookie_original": True, "total_pts": None, "position": "QB", "draft_number": 2},
        ]
    )
    monkeypatch.setattr(projector, "_project_rookies_with_adp", lambda draft_players_df, **kwargs: draft_players_df)

    result = projector.project_rookies(dataframe, method="adp", adp_filepath="adp.csv")

    assert result["total_pts"].notna().all()


def test_project_rookies_draft_mode_projects_only_rookies() -> None:
    """Verify draft mode keeps veterans and fills rookie totals from draft slot."""
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "is_rookie_original": False, "total_pts": 18.0, "position": "QB", "draft_number": 1},
            {"player_id": 2, "is_rookie_original": True, "total_pts": None, "position": "QB", "draft_number": 5},
        ]
    )

    result = RookieProjector().project_rookies(dataframe, method="draft")

    assert float(result.loc[result["player_id"] == 1, "total_pts"].iloc[0]) == 18.0 and result["total_pts"].notna().all()


def test_project_rookies_returns_original_frame_when_no_rookies() -> None:
    """Verify no-rookie inputs bypass projection logic."""
    dataframe = pd.DataFrame([{"player_id": 1, "is_rookie_original": False, "total_pts": 10.0, "position": "QB"}])

    assert RookieProjector().project_rookies(dataframe, method="draft").equals(dataframe)


def test_project_rookies_hybrid_falls_back_to_draft_slot_when_adp_missing(monkeypatch) -> None:
    """Verify hybrid mode falls back to draft-slot estimates when ADP leaves gaps."""
    projector = RookieProjector(adp_matcher=object())
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "is_rookie_original": False, "total_pts": 20.0, "position": "QB", "draft_number": 1},
            {"player_id": 2, "is_rookie_original": True, "total_pts": None, "position": "QB", "draft_number": 2},
        ]
    )
    monkeypatch.setattr(projector, "_project_rookies_with_adp", lambda draft_players_df, **kwargs: draft_players_df)
    result = projector.project_rookies(dataframe, method="hybrid", adp_filepath="adp.csv")

    assert result["total_pts"].notna().all()
