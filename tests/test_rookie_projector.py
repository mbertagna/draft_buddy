"""Tests for rookie projection behavior."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.rookie_projector import RookieProjector


def test_estimate_by_draft_slot_returns_zero_when_no_veterans() -> None:
    """Verify positions with no veteran history get zero projected points."""
    rookies_df = pd.DataFrame([{"position": "QB", "draft_number": 1}])
    veterans_df = pd.DataFrame(columns=["position", "total_pts"])
    result = RookieProjector().estimate_by_draft_slot(rookies_df, veterans_df)

    assert float(result.iloc[0]["total_pts"]) == 0.0


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
