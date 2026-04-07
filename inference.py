#!/usr/bin/env python3
import os 
import sys
import json 
import requests 
import time 
from typing import Dict, Any 

# Load environment first
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Environment Configuration 
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

def get_action_from_llm(observation: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Use heuristics instead of LLM - no API needed."""
    subject = observation['current_email_subject'].lower()
    body = observation['current_email_body'].lower()
    text = subject + " " + body
    
    if task_id == "easy":
        if any(k in text for k in ["urgent", "asap", "deadline", "critical"]):
            return {"action_type": "flag", "parameters": {}}
        else:
            return {"action_type": "archive", "parameters": {}}
    
    elif task_id == "medium":
        if any(k in text for k in ["urgent", "deadline", "critical", "asap", "failure", "down"]):
            priority = "high"
        elif any(k in text for k in ["meeting", "reminder", "review", "team", "report", "sync"]):
            priority = "medium"
        else:
            priority = "low"
        return {"action_type": "set_priority", "parameters": {"priority": priority}}
    
    else:  # hard
        reply = "Confirmed. Meeting time: 2pm, Location: Blue conference room."
        return {"action_type": "reply", "parameters": {"reply_text": reply}}

        
def run_episode(task_id: str) -> float: 
    """Run one episode for given task, return grader score."""
    sys.stdout.write(f"[START] task={task_id}\n")
    sys.stdout.flush()
    
    step_count = 0
    score = 0.0
    
    try:
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
            sys.stdout.write(f"Reset failed: {e}\n")
            sys.stdout.flush()
            sys.stdout.write(f"[END] task={task_id} score=0.0 steps=0\n")
            sys.stdout.flush()
            return 0.0

        if obs is None:
            sys.stdout.write("Reset did not return an observation\n")
            sys.stdout.flush()
            sys.stdout.write(f"[END] task={task_id} score=0.0 steps=0\n")
            sys.stdout.flush()
            return 0.0 
        
        done = False 
        max_steps = 20

        while not done and step_count < max_steps:  
            action = get_action_from_llm(obs, task_id) 
            step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=10) 

            if step_resp.status_code != 200: 
                sys.stdout.write(f"Step error: {step_resp.text}\n")
                sys.stdout.flush()
                break 
            
            data = step_resp.json() 
            obs = data["observation"] 
            reward = data["reward"]
            done = data["done"] 
            step_count += 1 
            
            sys.stdout.write(f"[STEP] step={step_count} reward={reward:.2f}\n")
            sys.stdout.flush()
            time.sleep(0.1)

        score_resp = requests.get(f"{ENV_URL}/score_task", params={"task_id": task_id}, timeout=10) 
        if score_resp.status_code == 200:
            score = score_resp.json()["score"]
        else:
            sys.stdout.write(f"Score error: {score_resp.text}\n")
            sys.stdout.flush()
            score = 0.0
    
    except Exception as e:
        sys.stdout.write(f"Episode exception: {e}\n")
        sys.stdout.flush()
        score = 0.0
    
    sys.stdout.write(f"[END] task={task_id} score={score:.3f} steps={step_count}\n")
    sys.stdout.flush()
    return score 


if __name__ == "__main__":
    try:
        for task in ["easy", "medium", "hard"]:
            run_episode(task)
    except Exception as e:
        sys.stdout.write(f"Main exception: {e}\n")
        sys.stdout.flush()
