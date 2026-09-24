"""Remaining-week Monte Carlo playoff-odds estimator."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from draft_buddy.simulator.evaluator import simulate_season_fast
from draft_buddy.simulator.team_identity import parse_team_id


class RemainingWeekMonteCarlo:
    """Sample noisy remaining-season outcomes and aggregate playoff odds.

    Parameters
    ----------
    roster_structure : mapping of str to int
        Starter slots including FLEX.
    num_playoff_teams : int
        Playoff cutoff size.
    noise_sigma : float
        Weekly noise as a fraction of projected points.
    rng : numpy.random.Generator, optional
        Random generator. Created when omitted.
    """

    def __init__(
        self,
        roster_structure: Mapping[str, int],
        num_playoff_teams: int,
        noise_sigma: float = 0.30,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._roster_structure = dict(roster_structure)
        self._num_playoff_teams = int(num_playoff_teams)
        self._noise_sigma = float(noise_sigma)
        self._rng = rng or np.random.default_rng()

    def worker_settings(self) -> Dict[str, Any]:
        """Return the fields a worker process needs to rebuild this estimator.

        Returns
        -------
        dict
            Roster structure, playoff cutoff, and weekly noise scale.
        """
        return {
            "roster_structure": dict(self._roster_structure),
            "num_playoff_teams": self._num_playoff_teams,
            "noise_sigma": self._noise_sigma,
        }

    def evaluate(
        self,
        weekly_projections: Mapping[int, Mapping[str, Any]],
        matchups: pd.DataFrame,
        rosters: Mapping[str, list[int]],
        n_runs: int,
        start_week: int = 1,
        starting_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
        season: int = 2026,
    ) -> Dict[int, Dict[str, float]]:
        """Estimate playoff probability and expected wins per team.

        Parameters
        ----------
        weekly_projections : mapping
            Simulator-shaped weekly projections.
        matchups : pandas.DataFrame
            Remaining regular-season schedule.
        rosters : mapping
            Player id lists keyed by ``Team {id}``.
        n_runs : int
            Number of noisy rollouts.
        start_week : int, optional
            First simulated week.
        starting_records : mapping, optional
            Locked W/L/T/pts by team label.
        season : int, optional
            Season year passed through to the evaluator.

        Returns
        -------
        dict of int to dict
            ``p_playoff`` and ``expected_wins`` keyed by team id.
        """
        labels = list(rosters)
        playoff_counts = {label: 0 for label in labels}
        win_totals = {label: 0.0 for label in labels}
        runs = max(1, int(n_runs))
        for _ in range(runs):
            noisy = self._perturb_remaining_weeks(weekly_projections, start_week)
            _results, records, _playoffs, _tree, _winner = simulate_season_fast(
                noisy,
                matchups,
                dict(rosters),
                season,
                num_playoff_teams=self._num_playoff_teams,
                start_week=start_week,
                starting_records=dict(starting_records or {}),
                roster_structure=self._roster_structure,
            )
            playoff_labels = {str(name) for name, _record in records[: self._num_playoff_teams]}
            for name, record in records:
                label = str(name)
                if label not in win_totals:
                    win_totals[label] = 0.0
                    playoff_counts.setdefault(label, 0)
                win_totals[label] += float(record["W"]) + 0.5 * float(record["T"])
                if label in playoff_labels:
                    playoff_counts[label] += 1
        return self._summarize(playoff_counts, win_totals, runs)

    def evaluate_once(
        self,
        weekly_projections: Mapping[int, Mapping[str, Any]],
        matchups: pd.DataFrame,
        rosters: Mapping[str, list[int]],
        start_week: int = 1,
        starting_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
        season: int = 2026,
    ) -> Dict[int, Dict[str, float]]:
        """Run one deterministic remaining-week simulation.

        Parameters
        ----------
        weekly_projections : mapping
            Simulator-shaped weekly projections.
        matchups : pandas.DataFrame
            Remaining regular-season schedule.
        rosters : mapping
            Player id lists keyed by ``Team {id}``.
        start_week : int, optional
            First simulated week.
        starting_records : mapping, optional
            Locked W/L/T/pts by team label.
        season : int, optional
            Season year passed through to the evaluator.

        Returns
        -------
        dict of int to dict
            ``p_playoff`` as 0/1 and ``expected_wins`` from the single run.
        """
        return self.evaluate(
            weekly_projections,
            matchups,
            rosters,
            n_runs=1,
            start_week=start_week,
            starting_records=starting_records,
            season=season,
        )

    def _perturb_remaining_weeks(
        self,
        weekly_projections: Mapping[int, Mapping[str, Any]],
        start_week: int,
    ) -> Dict[int, Dict[str, Any]]:
        """Copy weekly points and add Gaussian noise from ``start_week`` onward."""
        noisy: Dict[int, Dict[str, Any]] = {}
        start_index = max(0, int(start_week) - 1)
        for player_id, row in weekly_projections.items():
            pts = [float(value) for value in row.get("pts") or []]
            perturbed = list(pts)
            for index in range(start_index, len(perturbed)):
                mu = perturbed[index]
                if mu <= 0:
                    perturbed[index] = 0.0
                    continue
                draw = self._rng.normal(mu, self._noise_sigma * mu)
                perturbed[index] = float(max(0.0, draw))
            noisy[int(player_id)] = {"pos": row.get("pos") or row.get("position"), "pts": perturbed}
        return noisy

    def _summarize(
        self,
        playoff_counts: Mapping[str, int],
        win_totals: Mapping[str, float],
        runs: int,
    ) -> Dict[int, Dict[str, float]]:
        """Convert counts into per-team-id metrics."""
        summary: Dict[int, Dict[str, float]] = {}
        for label, count in playoff_counts.items():
            team_id = parse_team_id(str(label))
            if team_id is None:
                continue
            summary[team_id] = {
                "p_playoff": float(count) / float(runs),
                "expected_wins": float(win_totals.get(label, 0.0)) / float(runs),
            }
        return summary
