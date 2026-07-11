"""FastAPI entrypoint for draft operations."""

from __future__ import annotations

import asyncio
import csv
import io
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from draft_buddy.config import Config
from draft_buddy.data.insights.loader import LoadedPlayerInsights
from draft_buddy.simulator.service import SeasonSimulationService
from draft_buddy.web.advisor_factory import AdvisorGatewayRegistry
from draft_buddy.web.draft_advisor_schemas import AdvisorRequest
from draft_buddy.web.draft_advisor_service import DraftAdvisorError, DraftAdvisorService
from draft_buddy.web.session import DraftSessionManager


def create_app(
    config: Optional[Config] = None,
    session_manager: Optional[DraftSessionManager] = None,
    loaded_insights: Optional[LoadedPlayerInsights] = None,
    advisor_service: Optional[DraftAdvisorService] = None,
    advisor_registry: Optional[AdvisorGatewayRegistry] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    config : Optional[Config], optional
        Application configuration instance.
    session_manager : Optional[DraftSessionManager], optional
        Session manager instance, typically injected by composition root.
    loaded_insights : Optional[LoadedPlayerInsights], optional
        Pre-loaded player insights export for UI enrichment.
    advisor_service : Optional[DraftAdvisorService], optional
        Draft assistant service for structured LLM recommendations.
    advisor_registry : Optional[AdvisorGatewayRegistry], optional
        Registry exposing configured advisor models for the UI.

    Returns
    -------
    FastAPI
        Configured application object.
    """
    app = FastAPI(title="Draft Buddy API")
    runtime_config = config or Config()
    season_simulation_service = SeasonSimulationService(runtime_config)
    runtime_session_manager = session_manager or DraftSessionManager(runtime_config)
    frontend_index = Path(__file__).resolve().parents[3] / "frontend" / "index.html"
    insights_store = loaded_insights or LoadedPlayerInsights(
        players={}, meta=None, source_path=None
    )
    insights_by_id = insights_store.players
    insights_file_meta = insights_store.meta
    insights_source_path = insights_store.source_path
    runtime_advisor_service = advisor_service
    runtime_advisor_registry = advisor_registry

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Return HTTP errors in a consistent success/message envelope.

        The frontend treats every mutation response uniformly: successful
        responses carry the draft state, while failures carry a human-readable
        message. Centralizing the error shape here keeps individual endpoints
        free of response-formatting concerns.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "message": exc.detail},
        )

    def _session_id(request: Request, response: Optional[Response] = None) -> str:
        """Resolve session id from cookie or create one."""
        existing = request.cookies.get("draft_session_id")
        if existing:
            return existing
        new_id = str(uuid.uuid4())
        if response is not None:
            response.set_cookie("draft_session_id", new_id, httponly=True)
        return new_id


    @app.get("/api/hello")
    def hello_world() -> dict:
        """Return API health message."""
        return {"message": "Hello from DRAFT BUDDY backend!"}

    @app.get("/", response_class=FileResponse, response_model=None)
    def dashboard() -> Response:
        """Serve the dashboard frontend entry page."""
        if not frontend_index.exists():
            return PlainTextResponse(
                "Frontend entry file not found. Expected frontend/index.html.",
                status_code=500,
            )
        return FileResponse(frontend_index)


    @app.post("/api/draft/new")
    def create_new_draft(request: Request, response: Response) -> dict:
        """Create and return a fresh draft for current session."""
        session = runtime_session_manager.create_new(_session_id(request, response))
        return session.get_ui_state()


    @app.get("/api/draft/state")
    def draft_state(request: Request, response: Response) -> dict:
        """Return current session draft state."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        return session.get_ui_state()


    @app.post("/api/draft/pick")
    async def draft_pick(request: Request, response: Response) -> dict:
        """Apply a manual pick for current session."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        payload = await request.json()
        player_id = payload.get("player_id")
        if player_id is None:
            raise HTTPException(status_code=400, detail="Player ID is required")
        try:
            session.draft_player(int(player_id))
            session.save_state(runtime_config.paths.DRAFT_STATE_FILE)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.get_ui_state()


    @app.post("/api/draft/undo")
    def undo_pick(request: Request, response: Response) -> dict:
        """Undo most recent draft action."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        try:
            session.undo_last_pick()
            session.save_state(runtime_config.paths.DRAFT_STATE_FILE)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.get_ui_state()


    @app.post("/api/draft/transfer")
    async def transfer_player(request: Request, response: Response) -> dict:
        """Transfer one drafted player to another team or visual slot."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        payload = await request.json()
        player_id = payload.get("player_id")
        to_team_id = payload.get("to_team_id")
        to_round = payload.get("to_round")
        if player_id is None:
            raise HTTPException(status_code=400, detail="Player ID is required")
        if to_team_id is None:
            raise HTTPException(status_code=400, detail="Destination team ID is required")
        try:
            resolved_round = int(to_round) if to_round is not None else None
            session.transfer_player(int(player_id), int(to_team_id), to_round=resolved_round)
            session.save_state(runtime_config.paths.DRAFT_STATE_FILE)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.get_ui_state()


    @app.post("/api/draft/swap")
    async def swap_players(request: Request, response: Response) -> dict:
        """Swap two drafted players' teams and visual slots."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        payload = await request.json()
        player_id_1 = payload.get("player_id_1")
        player_id_2 = payload.get("player_id_2")
        if player_id_1 is None or player_id_2 is None:
            raise HTTPException(
                status_code=400, detail="Both player_id_1 and player_id_2 are required"
            )
        try:
            session.swap_players(int(player_id_1), int(player_id_2))
            session.save_state(runtime_config.paths.DRAFT_STATE_FILE)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.get_ui_state()


    @app.post("/api/draft/override_team")
    async def override_team(request: Request, response: Response) -> dict:
        """Override the next drafting team."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        payload = await request.json()
        team_id = payload.get("team_id")
        if team_id is None:
            raise HTTPException(status_code=400, detail="Team ID is required")
        try:
            session.set_current_team_picking(int(team_id))
            session.save_state(runtime_config.paths.DRAFT_STATE_FILE)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.get_ui_state()


    @app.post("/api/draft/simulate_pick")
    async def simulate_pick(request: Request, response: Response) -> dict:
        """Simulate one pick."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        use_policy = False
        try:
            payload = await request.json()
            if isinstance(payload, dict):
                use_policy = bool(payload.get("use_policy", False))
        except Exception:
            pass
        try:
            session.simulate_single_pick(use_policy=use_policy)
            session.save_state(runtime_config.paths.DRAFT_STATE_FILE)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.get_ui_state()


    @app.post("/api/draft/simulate_rest")
    async def simulate_rest(request: Request, response: Response) -> dict:
        """Simulate all remaining picks."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        use_policy = False
        try:
            payload = await request.json()
            if isinstance(payload, dict):
                use_policy = bool(payload.get("use_policy", False))
        except Exception:
            pass
        try:
            session.simulate_scheduled_picks_remaining(use_policy=use_policy)
            session.save_state(runtime_config.paths.DRAFT_STATE_FILE)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.get_ui_state()


    @app.get("/api/draft/ai_suggestion")
    def ai_suggestion(request: Request, response: Response) -> dict:
        """Return AI position probabilities for the team on the clock."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        return session.get_ai_suggestion()


    @app.get("/api/draft/ai_suggestions_all")
    def ai_suggestions_all(request: Request, response: Response) -> dict:
        """Return AI probabilities for all teams."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        return session.get_ai_suggestions_all()


    @app.get("/api/draft/ai_suggestion_for_team")
    def ai_suggestion_for_team(
        request: Request,
        response: Response,
        team_id: int = Query(...),
        ignore: Optional[str] = Query(default=None),
    ) -> dict:
        """Return AI probabilities for a specific team perspective."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        ignore_ids: List[int] = []
        if ignore:
            for token in ignore.split(","):
                stripped = token.strip()
                if stripped.isdigit():
                    ignore_ids.append(int(stripped))
        return session.get_ai_suggestion_for_team(team_id=team_id, ignore_player_ids=ignore_ids)


    @app.get("/api/draft/advisor/models")
    def draft_advisor_models() -> dict:
        """Return configured advisor models and UI defaults."""
        if runtime_advisor_registry is None:
            raise HTTPException(
                status_code=503,
                detail="Draft assistant is not configured. Set GEMINI_API_KEY and/or OPENROUTER_API_KEY.",
            )
        models = [
            {
                "id": option.id,
                "label": option.label,
                "provider": option.provider.value,
            }
            for option in runtime_advisor_registry.available_models()
        ]
        return {
            "models": models,
            "defaults": {
                "agent_model": runtime_advisor_registry.default_agent_model(),
                "other_teams_model": runtime_advisor_registry.default_other_teams_model(),
            },
        }

    @app.post("/api/draft/advisor")
    async def draft_advisor(request: Request, response: Response) -> dict:
        """Return a structured LLM draft recommendation for the advising team."""
        if runtime_advisor_service is None:
            raise HTTPException(
                status_code=503,
                detail="Draft assistant is not configured. Set GEMINI_API_KEY and/or OPENROUTER_API_KEY.",
            )
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        payload = await request.json()
        try:
            advisor_request = AdvisorRequest.model_validate(payload or {})
            recommendation = await asyncio.to_thread(
                runtime_advisor_service.recommend,
                session,
                advisor_request,
                insights_by_id,
            )
        except DraftAdvisorError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        return recommendation.model_dump(mode="json")


    @app.get("/api/draft/summary")
    def draft_summary(request: Request, response: Response) -> dict:
        """Return summary counts by position."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        summary = {
            "total_picks": session.current_pick_number - 1,
            "picks_by_position": {"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        }
        for pick in session.draft_history:
            player = session.player_catalog.get(pick.player_id)
            if player:
                summary["picks_by_position"][player.position] += 1
        return summary


    @app.get("/api/draft/export_csv")
    def export_csv(request: Request, response: Response) -> PlainTextResponse:
        """Export draft results as CSV content."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "Pick Number",
                "Team ID",
                "Player ID",
                "Name",
                "Position",
                "Team",
                "Bye Week",
                "Projected Points",
                "ADP",
                "Games Played %",
            ]
        )
        for pick in session.draft_history:
            player = session.player_catalog.get(pick.player_id)
            if player is None:
                continue
            writer.writerow(
                [
                    pick.pick_number,
                    pick.team_id,
                    player.player_id,
                    player.name,
                    player.position,
                    player.team or "N/A",
                    player.bye_week if player.bye_week and not np.isnan(player.bye_week) else "N/A",
                    player.projected_points,
                    player.adp if np.isfinite(player.adp) else "N/A",
                    player.games_played_frac,
                ]
            )
        return PlainTextResponse(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=draft_results.csv"},
        )


    @app.post("/api/simulate_season")
    def simulate_season(request: Request, response: Response) -> dict:
        """Run season simulation for current session rosters."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        try:
            return season_simulation_service.simulate_season(
                draft_state=session._state,
                player_catalog=session.player_catalog,
                team_manager_mapping=session.team_manager_mapping,
                weekly_projections=session.weekly_projections,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Season simulation failed: {error}") from error


    @app.get("/api/insights/meta")
    def insights_meta() -> dict:
        """Return metadata for the loaded player insights export."""
        if insights_file_meta is None or insights_source_path is None:
            return {
                "available": False,
                "draft_year": None,
                "generated_at": None,
                "model": None,
                "enriched_player_count": 0,
                "source_file": None,
            }
        return {
            "available": True,
            "draft_year": insights_file_meta.draft_year,
            "generated_at": insights_file_meta.generated_at.isoformat().replace("+00:00", "Z"),
            "model": insights_file_meta.model,
            "enriched_player_count": len(insights_by_id),
            "source_file": Path(insights_source_path).name,
        }


    @app.get("/api/players")
    def get_players(
        request: Request,
        response: Response,
        position: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "vorp",
        sort_dir: str = "desc",
    ) -> JSONResponse:
        """Return players filtered and sorted for frontend table."""
        session = runtime_session_manager.get_or_create(_session_id(request, response))
        filtered_players = session.player_catalog.resolve(session.available_player_ids)

        if position:
            positions = [value.strip().upper() for value in position.split(",")]
            filtered_players = [player for player in filtered_players if player.position in positions]
        if search:
            needle = search.lower()
            filtered_players = [player for player in filtered_players if needle in player.name.lower()]

        baselines = session.get_positional_baselines()
        player_vorp_map = {
            player.player_id: player.projected_points - baselines.get(player.position, 0.0)
            for player in filtered_players
        }

        reverse = sort_dir.lower() == "desc"

        def sort_key(player) -> tuple:
            if sort_by == "vorp":
                value = player_vorp_map.get(player.player_id, 0.0)
            elif sort_by == "adp":
                value = player.adp if np.isfinite(player.adp) else float("inf")
            elif sort_by == "projected_points":
                value = player.projected_points
            elif sort_by == "name":
                value = player.name.lower()
            elif sort_by == "position":
                value = player.position
            else:
                value = player_vorp_map.get(player.player_id, 0.0)
            return (value, player.player_id)

        filtered_players.sort(key=sort_key, reverse=reverse)
        payload = []
        for player in filtered_players:
            insight = insights_by_id.get(player.player_id)
            payload.append(
                {
                    "player_id": player.player_id,
                    "name": player.name,
                    "position": player.position,
                    "projected_points": player.projected_points,
                    "vorp": player_vorp_map.get(player.player_id, 0.0),
                    "games_played_frac": player.games_played_frac,
                    "adp": None if np.isinf(player.adp) else player.adp,
                    "bye_week": player.bye_week if player.bye_week and not np.isnan(player.bye_week) else "N/A",
                    "team": player.team,
                    "sleeper_status": player.sleeper_status,
                    "sleeper_injury_status": player.sleeper_injury_status,
                    "sleeper_depth_chart_position": player.sleeper_depth_chart_position,
                    "insight": insight.model_dump(mode="json") if insight else None,
                }
            )
        return JSONResponse(payload)

    return app


app = create_app()
