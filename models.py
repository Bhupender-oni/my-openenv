"""
Data models for Email Triage Environment.

Defines request/response schemas for the API.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    """Request model for taking an action."""
    action_type: str = Field(..., description="Type of action to take")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class ObservationResponse(BaseModel):
    """Response model containing environment observation."""
    current_email_subject: str
    current_email_body: str
    current_email_id: int
    email_index: int
    total_emails: int
    done: bool
    last_reward: float = 0.0


class RewardResponse(BaseModel):
    """Response model containing reward information."""
    value: float
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    """Response model for step action."""
    observation: ObservationResponse
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response model for environment state."""
    task_id: str
    current_idx: int
    done: bool
    total_reward: float
    action_history: list


class ScoreResponse(BaseModel):
    """Response model for grading."""
    task_id: str
    score: float


class TaskInfo(BaseModel):
    """Information about a task."""
    id: str
    name: str
    difficulty: str


class TaskListResponse(BaseModel):
    """Response model for task list."""
    tasks: list[TaskInfo]
