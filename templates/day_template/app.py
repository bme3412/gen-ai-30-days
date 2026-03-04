"""
30 Days of Agentic AI - Demo App Template
FastAPI backend for industry-vertical demo applications.

Usage:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Any, Optional
import json

from config import DAY_CONFIG, VERTICALS, get_vertical, get_all_verticals
from core.agent import run_agent

# Initialize FastAPI app
app = FastAPI(
    title=f"Day {DAY_CONFIG['day_number']}: {DAY_CONFIG['title']}",
    description="30 Days of Agentic AI Demo",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")


# Request/Response Models
class AgentRequest(BaseModel):
    vertical: str
    input: str
    sample_data: Optional[dict] = None


class AgentStep(BaseModel):
    type: str
    content: str


class AgentResponse(BaseModel):
    success: bool
    vertical: str
    steps: list[AgentStep]
    result: str
    error: Optional[str] = None


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main demo page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "day": DAY_CONFIG,
            "verticals": get_all_verticals()
        }
    )


@app.post("/api/run", response_model=AgentResponse)
async def run_agent_endpoint(request: AgentRequest):
    """
    Run the agent with the specified vertical and input.

    Args:
        request: Contains vertical ID, user input, and optional sample data

    Returns:
        AgentResponse with steps taken and final result
    """
    # Validate vertical
    vertical = get_vertical(request.vertical)
    if not vertical:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vertical: {request.vertical}"
        )

    try:
        # Run the day's core agent logic
        result = await run_agent(
            user_input=request.input,
            vertical_config=vertical,
            sample_data=request.sample_data or vertical.get("sample_data", {})
        )

        return AgentResponse(
            success=True,
            vertical=request.vertical,
            steps=result.get("steps", []),
            result=result.get("result", "No result generated")
        )

    except Exception as e:
        return AgentResponse(
            success=False,
            vertical=request.vertical,
            steps=[],
            result="",
            error=str(e)
        )


@app.get("/api/verticals")
async def get_verticals():
    """Get all available industry verticals."""
    return {
        "verticals": get_all_verticals()
    }


@app.get("/api/vertical/{vertical_id}")
async def get_vertical_info(vertical_id: str):
    """Get information about a specific vertical."""
    vertical = get_vertical(vertical_id)
    if not vertical:
        raise HTTPException(
            status_code=404,
            detail=f"Vertical not found: {vertical_id}"
        )
    return vertical


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "day": DAY_CONFIG["day_number"],
        "title": DAY_CONFIG["title"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
