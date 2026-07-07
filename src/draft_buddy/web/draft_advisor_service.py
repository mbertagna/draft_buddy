"""Orchestration for the live draft assistant."""

from __future__ import annotations

from typing import Dict, Optional

from draft_buddy.data.insights.schemas import PlayerInsight
from draft_buddy.web.advisor_factory import AdvisorGatewayRegistry
from draft_buddy.web.draft_advisor_context import (
    DEFAULT_TOP_K,
    POSITIONS,
    RECENT_PICKS_LIMIT,
    build_advisor_context,
    build_candidate_rows,
    collect_candidate_player_ids,
    filter_available_players,
    position_top_k_map,
)
from draft_buddy.web.draft_advisor_schemas import (
    AdvisorRequest,
    AdvisorResult,
    AdvisorScope,
    AdvisorTrigger,
)
from draft_buddy.web.draft_advisor_resilience import recommend_with_resilience
from draft_buddy.web.session import DraftSession

SYSTEM_PROMPT = (
    "You are a fantasy football draft analyst. Recommend exactly one player from the "
    "candidate tables in the user message. Ground your reasoning in the provided stats "
    "and offline insights. Never recommend a player who is not listed. "
    "Always populate rationale_bullets with 2-5 concise, evidence-based bullets. "
    "Include up to 3 alternates from the candidate tables when useful."
)


class DraftAdvisorError(Exception):
    """Base error for draft assistant failures."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class DraftAdvisorScopeError(DraftAdvisorError):
    """Raised when auto-fire scope rules block the request."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=403)


class DraftAdvisorValidationError(DraftAdvisorError):
    """Raised when the request or candidate pool is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class DraftAdvisorService:
    """Build context and request a structured draft recommendation."""

    def __init__(self, registry: AdvisorGatewayRegistry) -> None:
        """
        Parameters
        ----------
        registry : AdvisorGatewayRegistry
            Registry of configured advisor LLM gateways.
        """
        self._registry = registry

    def recommend(
        self,
        session: DraftSession,
        request: AdvisorRequest,
        insights: Dict[int, PlayerInsight],
    ) -> AdvisorResult:
        """Return a pick recommendation for the advising team.

        Parameters
        ----------
        session : DraftSession
            Active draft session.
        request : AdvisorRequest
            Assistant request parameters.
        insights : Dict[int, PlayerInsight]
            Offline player insights keyed by player id.

        Returns
        -------
        AdvisorResult
            Structured or degraded recommendation from the LLM.

        Raises
        ------
        DraftAdvisorError
            When scope, validation, or LLM calls fail.
        """
        ui_state = session.get_ui_state()
        advising_team_id = request.team_id or ui_state.get("current_team_picking")
        if advising_team_id is None:
            raise DraftAdvisorValidationError("No team is currently on the clock.")
        if not (1 <= advising_team_id <= session.num_teams):
            raise DraftAdvisorValidationError(f"Invalid team id {advising_team_id}.")

        self._enforce_scope(request, ui_state, advising_team_id, session.agent_team_id)
        self._validate_models(request)

        available_players = [
            session.player_catalog.require(player_id)
            for player_id in session.available_player_ids
        ]
        filtered_players = filter_available_players(available_players, request.gp_min)
        if not filtered_players:
            raise DraftAdvisorValidationError("No players pass your GP filter.")

        baselines = session.get_positional_baselines()
        candidate_rows = build_candidate_rows(filtered_players, baselines)
        rl_probs, rl_degraded = self._resolve_rl_probs(session, advising_team_id)
        top_k_by_position = (
            {position: DEFAULT_TOP_K for position in POSITIONS}
            if rl_degraded
            else position_top_k_map(rl_probs)
        )
        valid_player_ids = collect_candidate_player_ids(candidate_rows, top_k_by_position)
        if not valid_player_ids:
            raise DraftAdvisorValidationError("No candidates remain after building shortlists.")

        context = build_advisor_context(
            ui_state=ui_state,
            advising_team_id=advising_team_id,
            agent_team_id=session.agent_team_id,
            roster_structure=session.roster_structure,
            bench_maxes=session.bench_maxes,
            total_bench_size=session.total_bench_size,
            team_manager_mapping=session.team_manager_mapping,
            candidate_rows=candidate_rows,
            baselines=baselines,
            top_k_by_position=top_k_by_position,
            rl_probs=rl_probs,
            rl_degraded=rl_degraded,
            insights=insights,
            recent_picks=self._build_recent_pick_summaries(session),
        )

        model_id = self._resolve_model_id(request, advising_team_id, session.agent_team_id)
        gateway = self._registry.get(model_id)

        try:
            result = recommend_with_resilience(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=context,
                fetch_payload=lambda prompt: gateway.fetch_payload(SYSTEM_PROMPT, prompt),
                advising_team_id=advising_team_id,
                is_agent_team=advising_team_id == session.agent_team_id,
                valid_player_ids=valid_player_ids,
            )
        except Exception as error:
            raise DraftAdvisorError(
                f"Assistant request failed: {error}. Try the RL position chips instead.",
                status_code=502,
            ) from error

        return result

    def _build_recent_pick_summaries(self, session: DraftSession) -> list[dict[str, object]]:
        """Return recent draft picks with player names for advisor context."""
        recent_picks = session.draft_history[-RECENT_PICKS_LIMIT:]
        summaries: list[dict[str, object]] = []
        for pick in recent_picks:
            player = session.player_catalog.get(pick.player_id)
            summaries.append(
                {
                    "pick_number": pick.pick_number,
                    "team_id": pick.team_id,
                    "player_name": player.name if player is not None else f"Player {pick.player_id}",
                    "position": player.position if player is not None else "?",
                }
            )
        return summaries

    def _validate_models(self, request: AdvisorRequest) -> None:
        """Validate requested model ids against configured availability."""
        try:
            self._registry.validate_model(request.agent_model)
            self._registry.validate_model(request.other_teams_model)
        except ValueError as error:
            raise DraftAdvisorValidationError(str(error)) from error

    def _resolve_model_id(
        self,
        request: AdvisorRequest,
        advising_team_id: int,
        agent_team_id: int,
    ) -> str:
        """Return the model id for the advising team."""
        if advising_team_id == agent_team_id:
            return request.agent_model
        return request.other_teams_model

    def _enforce_scope(
        self,
        request: AdvisorRequest,
        ui_state: dict,
        advising_team_id: int,
        agent_team_id: int,
    ) -> None:
        """Enforce auto-fire scope rules."""
        if request.trigger != AdvisorTrigger.AUTO:
            return
        if request.scope == AdvisorScope.ALL_TEAMS:
            return
        snake_team = ui_state.get("snake_team_on_turn")
        if snake_team != agent_team_id:
            raise DraftAdvisorScopeError(
                "Auto assistant is limited to your picks only for this snake turn."
            )
        if advising_team_id != agent_team_id:
            raise DraftAdvisorScopeError(
                "Auto assistant can only advise your team when scope is agent_only."
            )

    def _resolve_rl_probs(
        self,
        session: DraftSession,
        advising_team_id: int,
    ) -> tuple[Dict[str, float], bool]:
        """Return RL position probabilities and whether fallback mode was used."""
        suggestion = session.get_ai_suggestion_for_team(advising_team_id)
        if suggestion.get("error"):
            fallback = {position: 0.25 for position in POSITIONS}
            return fallback, True
        probs = {position: float(suggestion.get(position, 0.0)) for position in POSITIONS}
        return probs, False
