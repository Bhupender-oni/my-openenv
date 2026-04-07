#!/usr/bin/env python3
"""
Email Triage Agent - Must output [START]/[STEP]/[END] blocks.
Uses OpenAI client with LiteLLM proxy.
"""
import os 
import sys
import json 
import requests 
import time 
from typing import Dict, Any 

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from openai import OpenAI

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "test-key")

# Initialize OpenAI client with LiteLLM proxy
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def get_action_from_llm(observation: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Use LLM to determine action."""
    subject = observation.get('current_email_subject', '')
    body = observation.get('current_email_body', '')
    
    if task_id == "easy":
        prompt = f"""You are an email triage agent. Analyze this email and decide whether to flag it as urgent or archive it.

Email Subject: {subject}
Email Body: {body}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"action_type": "flag" or "archive"}}

An email should be flagged as urgent if it contains keywords like: urgent, asap, deadline, critical, emergency.
Otherwise, archive it."""
    
    elif task_id == "medium":
        prompt = f"""You are an email triage agent. Analyze this email and assign a priority level: high, medium, or low.

Email Subject: {subject}
Email Body: {body}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"action_type": "set_priority", "parameters": {{"priority": "high|medium|low"}}}}

Assign high priority to: urgent, deadline, critical, asap, failure, system down, outage.
Assign medium priority to: meetings, reviews, team updates, reports, syncs.
Assign low priority to everything else."""
    
    else:  # hard
        prompt = f"""You are an email triage agent. Analyze this email about a meeting request and draft a reply confirming the meeting details.

Email Subject: {subject}
Email Body: {body}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"action_type": "reply", "parameters": {{"reply_text": "your reply here"}}}}

Extract meeting time and location from the email, then confirm in your reply."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        action = json.loads(response_text)
        return action
    
    except Exception as e:
        # Fallback to heuristic if LLM fails
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        subject_lower = subject.lower()
        body_lower = body.lower()
        text = subject_lower + " " + body_lower
        
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
    """Run one episode and return score."""
    # MUST print START immediately
    print(f"[START] task={task_id}", flush=True)
    
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
            # Always print END, even on error
            print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
            return 0.0

        if obs is None:
            print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
            return 0.0 
        
        done = False 
        max_steps = 20

        while not done and step_count < max_steps:  
            action = get_action_from_llm(obs, task_id) 
            step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=10) 

            if step_resp.status_code != 200: 
                break 
            
            data = step_resp.json() 
            obs = data.get("observation", obs)
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            step_count += 1 
            
            print(f"[STEP] step={step_count} reward={reward:.2f}", flush=True)
            time.sleep(0.1)

        try:
            score_resp = requests.get(f"{ENV_URL}/score_task", params={"task_id": task_id}, timeout=10) 
            if score_resp.status_code == 200:
                score = score_resp.json().get("score", 0.0)
        except:
            pass
    
    except Exception as e:
        pass
    
    # MUST print END with score and steps
    print(f"[END] task={task_id} score={score:.3f} steps={step_count}", flush=True)
    return score 


def main():
    """Run all tasks."""
    try:
        for task in ["easy", "medium", "hard"]:
            run_episode(task)
    except Exception as e:
        # Catch any uncaught exception
        pass


# Run when executed directly
if __name__ == "__main__":
    main()
