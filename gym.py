"""
Gym-compatible wrapper for Email Triage Environment.

Provides compatibility with OpenAI Gym/Gymnasium API.
"""

from typing import Tuple, Dict, Any
from email_env import EmailTriageEnv, Action, ActionType, Observation, Reward


class EmailTriageGymEnv:
    """Gym-compatible wrapper for email triage environment."""
    
    def __init__(self, task_id: str = "easy"):
        """Initialize environment."""
        self.env = EmailTriageEnv(task_id=task_id)
        self.task_id = task_id
    
    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        return self.env.reset(self.task_id)
    
    def step(self, action: Dict[str, Any]) -> Tuple[Observation, float, bool, Dict]:
        """Execute action and return observation, reward, done, info."""
        # Convert dict to Action object
        action_obj = Action(
            action_type=action["action_type"],
            parameters=action.get("parameters", {})
        )
        obs, reward, done, info = self.env.step(action_obj)
        return obs, reward.value, done, info
    
    def render(self, mode: str = "human") -> None:
        """Render environment state."""
        state = self.env.state()
        print(f"Task: {state['task_id']}")
        print(f"Current Index: {state['current_idx']}")
        print(f"Done: {state['done']}")
        print(f"Total Reward: {state['total_reward']:.2f}")
    
    def close(self) -> None:
        """Close environment."""
        pass
