"""
Gradio web interface for Email Triage Environment
"""

import gradio as gr
import requests
import json
from typing import Tuple

ENV_URL = "http://localhost:8000"

def reset_env(task_id: str):
    """Reset the environment."""
    resp = requests.get(f"{ENV_URL}/reset", params={"task_id": task_id})
    data = resp.json()
    return (
        data["current_email_subject"],
        data["current_email_body"],
        json.dumps(data, indent=2),
        0.0
    )

def flag_email():
    """Flag current email as urgent."""
    action = {"action_type": "flag", "parameters": {}}
    resp = requests.post(f"{ENV_URL}/step", json=action)
    data = resp.json()
    return (
        data["observation"]["current_email_subject"],
        data["observation"]["current_email_body"],
        data["reward"],
        data["done"]
    )

def archive_email():
    """Archive current email."""
    action = {"action_type": "archive", "parameters": {}}
    resp = requests.post(f"{ENV_URL}/step", json=action)
    data = resp.json()
    return (
        data["observation"]["current_email_subject"],
        data["observation"]["current_email_body"],
        data["reward"],
        data["done"]
    )

def set_priority(priority: str):
    """Set priority for current email."""
    action = {"action_type": "set_priority", "parameters": {"priority": priority}}
    resp = requests.post(f"{ENV_URL}/step", json=action)
    data = resp.json()
    return (
        data["observation"]["current_email_subject"],
        data["observation"]["current_email_body"],
        data["reward"],
        data["done"]
    )

def reply_email(reply_text: str):
    """Reply to email."""
    action = {"action_type": "reply", "parameters": {"reply_text": reply_text}}
    resp = requests.post(f"{ENV_URL}/step", json=action)
    data = resp.json()
    return (
        data["observation"]["current_email_subject"],
        data["observation"]["current_email_body"],
        data["reward"],
        data["done"]
    )

def get_score(task_id: str):
    """Get grader score."""
    resp = requests.get(f"{ENV_URL}/score_task", params={"task_id": task_id})
    score = resp.json()["score"]
    return f"Score: {score:.3f}"

# Create Gradio interface
with gr.Blocks(title="Email Triage Environment") as demo:
    gr.Markdown("# Email Triage Environment")
    gr.Markdown("Train AI agents to triage emails across 3 difficulty levels")
    
    with gr.Row():
        task_selector = gr.Radio(
            ["easy", "medium", "hard"],
            value="easy",
            label="Select Task"
        )
        reset_btn = gr.Button("Reset Task")
    
    with gr.Row():
        subject = gr.Textbox(label="Email Subject", interactive=False)
        body = gr.Textbox(label="Email Body", interactive=False, lines=4)
    
    with gr.Row():
        reward = gr.Number(label="Reward", interactive=False)
        done = gr.Checkbox(label="Task Complete", interactive=False)
    
    # Easy task controls
    with gr.Group(visible=True) as easy_group:
        gr.Markdown("**Easy Task**: Flag urgent emails")
        with gr.Row():
            flag_btn = gr.Button("Flag as Urgent")
            archive_btn = gr.Button("Archive")
    
    # Medium task controls
    with gr.Group(visible=False) as medium_group:
        gr.Markdown("**Medium Task**: Assign priority levels")
        with gr.Row():
            priority = gr.Radio(
                ["high", "medium", "low"],
                value="low",
                label="Priority"
            )
            set_priority_btn = gr.Button("Set Priority")
    
    # Hard task controls
    with gr.Group(visible=False) as hard_group:
        gr.Markdown("**Hard Task**: Extract meeting info and reply")
        reply_text = gr.Textbox(
            label="Reply Message",
            placeholder="Type your reply...",
            lines=3
        )
        reply_btn = gr.Button("Send Reply")
    
    with gr.Row():
        score_btn = gr.Button("Get Score")
        score_output = gr.Textbox(label="Score", interactive=False)
    
    state = gr.State({
        "subject": "",
        "body": "",
        "reward": 0.0,
        "done": False,
        "task_id": "easy"
    })
    
    # Update visibility based on task selection
    def update_visibility(task):
        return (
            gr.Group(visible=task == "easy"),
            gr.Group(visible=task == "medium"),
            gr.Group(visible=task == "hard")
        )
    
    task_selector.change(
        update_visibility,
        inputs=task_selector,
        outputs=[easy_group, medium_group, hard_group]
    )
    
    # Reset task
    def reset_task(task_id):
        result = reset_env(task_id)
        return {
            "subject": result[0],
            "body": result[1],
            "reward": 0.0,
            "done": False,
            "task_id": task_id
        }, result[0], result[1], 0.0, False
    
    reset_btn.click(
        reset_task,
        inputs=task_selector,
        outputs=[state, subject, body, reward, done]
    )
    
    # Easy task actions
    flag_btn.click(
        flag_email,
        outputs=[subject, body, reward, done]
    )
    archive_btn.click(
        archive_email,
        outputs=[subject, body, reward, done]
    )
    
    # Medium task action
    set_priority_btn.click(
        set_priority,
        inputs=priority,
        outputs=[subject, body, reward, done]
    )
    
    # Hard task action
    reply_btn.click(
        reply_email,
        inputs=reply_text,
        outputs=[subject, body, reward, done]
    )
    
    # Score button
    score_btn.click(
        get_score,
        inputs=task_selector,
        outputs=score_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
