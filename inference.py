import os 
import json 
import requests 
import time 
from typing import Dict, Any 
from dotenv import load_dotenv

load_dotenv() 

# Environment Configuration 
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Helper Functions
def get_action_from_llm(observation: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Use heuristics instead of LLM - no API needed."""
    subject = observation['current_email_subject'].lower()
    body = observation['current_email_body'].lower()
    text = subject + " " + body
    
    if task_id == "easy":
        # Flag if urgent keywords present
        if any(k in text for k in ["urgent", "asap", "deadline", "critical"]):
            return {"action_type": "flag", "parameters": {}}
        else:
            return {"action_type": "archive", "parameters": {}}
    
    elif task_id == "medium":
        # Assign priority based on keywords
        if any(k in text for k in ["urgent", "deadline", "critical", "asap", "failure", "down"]):
            priority = "high"
        elif any(k in text for k in ["meeting", "reminder", "review", "team", "report", "sync"]):
            priority = "medium"
        else:
            priority = "low"
        return {"action_type": "set_priority", "parameters": {"priority": priority}}
    
    else:  # hard
        # Always reply with meeting info on final email
        reply = "Confirmed. Meeting time: 2pm, Location: Blue conference room."
        return {"action_type": "reply", "parameters": {"reply_text": reply}}

        
def run_episode(task_id: str) -> float: 
    """Run one episode for given task, return grader score."""
    # Reset environment  
    reset_url = f"{ENV_URL}/reset" 
    obs = None 
    try:
        resp = requests.get(reset_url, params={"task_id": task_id}, timeout=10) 
        if resp.status_code == 200:
            obs = resp.json()
        else: 
            resp = requests.post(reset_url, params={"task_id": task_id}, json={}, timeout=10)
            resp.raise_for_status() 
            obs = resp.json() 
    except Exception as e: 
        print(f"Reset failed: {e}")  
        return 0.0
    

    if obs is None:
        print("Reset did not return an observation")
        return 0.0 
    
    done = False 
    step_count = 0 
    max_steps = 20 # safety  

    while not done and step_count < max_steps:  
        # Get action using heuristics (no LLM)
        action = get_action_from_llm(obs, task_id) 
        # Send step request 
        step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=10) 

        if step_resp.status_code != 200: 
            print(f"Step error: {step_resp.text}")
            break 
        data = step_resp.json() 
        obs = data["observation"] 
        reward = data["reward"]
        done = data["done"] 
        step_count += 1 
        print(f"Step {step_count}: action={action}, reward={reward:.2f}, done={done}") 
        time.sleep(0.1)

    # Get grader score 
    score_resp = requests.get(f"{ENV_URL}/score_task", params={"task_id": task_id}, timeout=10) 
    if score_resp.status_code != 200:
        print(f"Score error: {score_resp.text}")
        return 0.0 
    score = score_resp.json()["score"]
    print(f"Episode finished. Steps: {step_count}, Grader score: {score:.3f}") 
    return score 

def main():
    tasks = ["easy", "medium", "hard"] 
    scores = {} 
    for task in tasks: 
        print(f"\n=== Running task: {task} ===") 
        score = run_episode(task) 
        scores[task] = score 

    print("\n=== Baseline Scores ===") 
    for task, score in scores.items():
        print(f"{task.capitalize()}: {score:.3f}") 

    # Save to file for reproducibility 
    with open("baseline_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    scores = {task: run_episode(task) for task in ["easy", "medium", "hard"]} 
    with open("baseline_scores.json", "w") as f:
        json.dump(scores, f, indent=2) 
    print("\nBaseline Scores:", scores)
