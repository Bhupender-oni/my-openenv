"""
Email Triage Environment Client

Provides the client interface for interacting with the email triage environment.
"""

from typing import Dict, Any, Optional
import requests
from email_env import Action, ActionType


class EmailTriageClient:
    """Client for interacting with Email Triage Environment."""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        """Initialize client with server URL."""
        self.base_url = base_url
    
    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        """Reset environment to a new task."""
        resp = requests.get(f"{self.base_url}/reset", params={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Take a step in the environment."""
        resp = requests.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()
    
    def state(self) -> Dict[str, Any]:
        """Get current environment state."""
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()
    
    def score_task(self, task_id: str) -> float:
        """Get grader score for a task."""
        resp = requests.get(f"{self.base_url}/score_task", params={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()["score"]
    
    def list_tasks(self) -> Dict[str, Any]:
        """List available tasks."""
        resp = requests.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()
