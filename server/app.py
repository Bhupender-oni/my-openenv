import os
import sys
from pathlib import Path

# Add parent directory to path to import email_env
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
import uvicorn
from email_env import EmailTriageEnv, Action, GRADERS

app = FastAPI(title="Email Triage Environment", description="OpenEnv compliant email triage simulation")

# Global environment instance
env = None


def get_env():
    """Lazy initialization of environment."""
    global env
    if env is None:
        env = EmailTriageEnv(task_id="easy")
    return env


@app.get("/reset")
@app.post("/reset")
def reset(task_id: str = "easy"):
    """ Reset the environment to a new task (easy/medium/hard). """
    if task_id not in GRADERS:
        raise HTTPException(400, f"Unknown task. Choose from {list(GRADERS.keys())}")
    global env 
    env = EmailTriageEnv(task_id=task_id) 
    obs = env.reset(task_id)
    return obs.dict()


@app.post("/step")
def step(action: Action):
    """ Take an action on the current email."""
    env = get_env()
    if env is None: 
        raise HTTPException(400, "Environment not initialized. Call /reset first.")
    obs, reward, done, info = env.step(action) 
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    """ Return full internal state (for debugging)."""
    env = get_env()
    if env is None:
        raise HTTPException(400, "Environment not initialized.")
    return env.state()


@app.get("/score_task")
def score_task(task_id: str):
    """Run the grader for the given task and return score (0.0-1.0) """
    if task_id not in GRADERS:
        raise HTTPException(400, f"Unknown task: {task_id}")
    env = get_env()
    if env is None or env.task_id != task_id:
        # If environment is not set or different task, create temporary one
        temp_env = EmailTriageEnv(task_id=task_id) 
        temp_env.reset()
        # Note: This will only work if the agent has not run;  in practice, call after episode.
        # For proper scoring, run the episode with the same env.
        score = GRADERS[task_id](temp_env) 
        return {"task_id": task_id, "score": score}
    score = GRADERS[task_id](env) 
    return {"task_id": task_id, "score": score}


@app.get("/tasks")
def list_tasks(): 
    """Return available tasks and their difficulty."""
    return {
        "tasks": [
            {"id": "easy", "name": "Flag Urgent Emails", "difficulty": "easy"},
            {"id": "medium", "name": "Assign Priority Levels", "difficulty": "medium"},
            {"id": "hard", "name": "Conversation Reply", "difficulty": "hard"}
        ]
    }


def main():
    """Entry point for the server."""
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
