import numpy as np
from typing import Dict, Any, List

class DraftPresentationService:
    """
    Assembles complex draft state representations for UI presentation.
    """

    @staticmethod
    def get_ui_state(draft_env) -> Dict[str, Any]:
        """
        Builds a comprehensive dictionary of the current draft state.
        """
        team_on_clock = (
            draft_env._overridden_team_id
            if draft_env._overridden_team_id is not None
            else (
                draft_env.draft_order[draft_env.current_pick_idx]
                if draft_env.current_pick_idx < len(draft_env.draft_order)
                else None
            )
        )

        structured_rosters = {}
        team_points_summary = {}
        team_is_full = {}

        from draft_buddy.utils.roster_utils import categorize_roster_by_slots, calculate_roster_scores

        for team_id, roster_data in draft_env.teams_rosters.items():
            # Roster Categorization
            starters, bench, _ = categorize_roster_by_slots(
                roster_data['PLAYERS'],
                draft_env.roster_structure,
                draft_env.bench_maxes
            )
            
            # Points Summary
            scores = calculate_roster_scores(
                roster_data['PLAYERS'],
                draft_env.roster_structure,
                draft_env.bench_maxes
            )

            team_points_summary[team_id] = {
                'starters_total': scores['starters_total_points'],
                'bench_total': scores['bench_total_points']
            }

            structured_rosters[team_id] = {
                'starters': {pos: [p.to_dict() for p in players] for pos, players in starters.items()},
                'bench': [p.to_dict() for p in bench]
            }

            # Full status
            team_is_full[team_id] = (len(roster_data['PLAYERS']) >= draft_env.total_roster_size_per_team)

        return {
            'draft_order': draft_env.draft_order,
            'current_pick_number': draft_env.current_pick_number,
            'current_team_picking': team_on_clock,
            'team_rosters': structured_rosters,
            'roster_counts': {
                team_id: {pos: roster_data[pos] for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']} 
                for team_id, roster_data in draft_env.teams_rosters.items()
            },
            'team_projected_points': {
                team_id: sum(p.projected_points for p in roster_data['PLAYERS']) 
                for team_id, roster_data in draft_env.teams_rosters.items()
            },
            'manual_draft_teams': list(draft_env.manual_draft_teams),
            'roster_structure': draft_env.roster_structure,
            'team_is_full': team_is_full,
            'team_points_summary': team_points_summary,
            'num_teams': draft_env.num_teams,
            'team_bye_weeks': DraftPresentationService._aggregate_bye_weeks(draft_env),
            'agent_start_position': draft_env.agent_team_id,
        }

    @staticmethod
    def _aggregate_bye_weeks(draft_env) -> Dict[int, Dict[int, Dict[str, int]]]:
        """Calculates bye week aggregates for each team."""
        bye_data = {}
        for team_id, roster_data in draft_env.teams_rosters.items():
            players = roster_data['PLAYERS']
            team_byes = {}
            # Filter valid bye weeks
            weeks = set(p.bye_week for p in players if p.bye_week and not np.isnan(p.bye_week))
            for week in weeks:
                pos_counts = {}
                for p in players:
                    if p.bye_week == week:
                        pos_counts[p.position] = pos_counts.get(p.position, 0) + 1
                team_byes[int(week)] = pos_counts
            bye_data[team_id] = team_byes
        return bye_data
