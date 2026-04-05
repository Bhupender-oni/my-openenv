# email_env.py 
import re
from typing import List, Dict, Any, Optional 
from pydantic import BaseModel, Field 
from enum import Enum 

# Open Env Models
class ActionType(str, Enum):
    """Allowed actions in the environment."""
    SET_PRIORITY = "set_priority"
    FLAG = "flag"
    ARCHIVE = "archive"
    DELETE = "delete"
    REPLY = "reply"

class Action(BaseModel):
    """Action taken by the agent."""
    action_type: ActionType 
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    """Observation returned after each step."""
    current_email_subject: str
    current_email_body: str
    current_email_id: int
    email_index: int         # 0-index position in the inbox
    total_emails: int
    done: bool 
    last_reward: float = 0.0 # reward from the last step (for convenience)

class Reward(BaseModel):
    """Reward returned after a step."""
    value: float 
    info: Dict[str, Any] = Field(default_factory=dict)

# Email Data for Easy Task
EASY_EMAILS = [
    {"id": 0, "subject": "URGENT: Server down", "body": "Production server is not responding. Please fix ASAP.", "true_priority": "high"},
    {"id": 1, "subject": "Weekly report", "body": "Attached is the weekly sales report.", "true_priority": "low"},
    {"id": 2, "subject": "Meeting reminder", "body": "Reminder: team sync at 2pm.", "true_priority": "medium"},
    {"id": 3, "subject": "ASAP: Client request", "body": "Client needs the proposal by end of day.", "true_priority": "high"},
    {"id": 4, "subject": "Newsletter", "body": "Check out our latest product updates.", "true_priority": "low"},
]

MEDIUM_EMAILS = [
    {"id": 0, "subject": "Deadline extension", "body": "Can we push the deadline to Friday?", "true_priority": "high"},
    {"id": 1, "subject": "Lunch plans", "body": "Where should we eat today?", "true_priority": "low"},
    {"id": 2, "subject": "Bug report #4321", "body": "Login fails when password contains special chars.", "true_priority": "high"},
    {"id": 3, "subject": "Team building event", "body": "Sign up for the escape room by Thursday.", "true_priority": "medium"}, 
    {"id": 4, "subject": "Spam: Win a prize", "body": "Congratulations! You've won a free iPhone.", "true_priority": "low"},
    {"id": 5, "subject": "Quarterly review", "body": "Please prepare your Q2 metrics.", "true_priority": "medium"},
    {"id": 6, "subject": "Critical: Payment failure", "body": "Customer's payment failed repeatedly.", "true_priority": "high"},
    {"id": 7, "subject": "Coffee chat", "body": "Want to grab coffee later?", "true_priority": "low"},
]

HARD_EMAILS = [
    {"id": 0, "subject": "Meeting request", "body": "Hi, can we meet tomorrow at 2pm  to discuss the project? - Alice", "thread_id": "thread_1"},
    {"id": 1, "subject": "Re: Meeting request", "body": "2pm works for me. Where shall we meet? - Bob", "thread_id": "thread_1"},
    {"id": 2, "subject": "Re: Meeting request", "body": "Let's use the Blue conference room. See you at 2pm. - Alice", "thread_id": "thread_1"},
    {"id": 3, "subject": "Final confirmation", "body": "Please reply with 'Confirmed' and the time and location.", "thread_id": "thread_1"},
]
EXPECTED_EXTRACTION = {"time": "2pm", "location": "Blue conference room"}
  
# Email Environment
class EmailTriageEnv: 
    def __init__(self, task_id: str = "easy"): 
        self.task_id = task_id 
        self.emails = []
        self.current_idx = 0
        self.done = False 
        self.total_reward = 0.0
        self.action_history = []
        self.reset(task_id)  # initial reset

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset the environment to the beginning of the task."""
        if task_id:
            self.task_id = task_id 
        # For now only support "easy" 
        if self.task_id == "easy": 
            self.emails = EASY_EMAILS.copy()
        elif self.task_id == "medium":
            self.emails = MEDIUM_EMAILS.copy()
        elif self.task_id == "hard":
            self.emails = HARD_EMAILS.copy()
        else: 
            raise ValueError(f"Unknown task: {self.task_id}")
        self.current_idx = 0 
        self.done = False 
        self.total_reward = 0.0 
        self.action_history = [] 
        return self._get_obs() 


    def _get_obs(self) -> Observation:
        """Create the current observation from internal state."""
        if self.current_idx >= len(self.emails):
            return Observation(
                current_email_subject="",
                current_email_body="",
                current_email_id=-1,
                email_index=len(self.emails),
                total_emails=len(self.emails),
                done=True, 
                last_reward=0.0
            )
        email = self.emails[self.current_idx]
        return Observation(
            current_email_subject=email["subject"],
            current_email_body=email["body"],
            current_email_id=email["id"],
            email_index=self.current_idx,
            total_emails=len(self.emails),
            done=False,
            last_reward=0.0
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict]:
        """Process one action on the current email."""
        if self.done:
            return self._get_obs(), Reward(value=0.0), True, {}

        email = self.emails[self.current_idx]
        reward_val = 0.0
        info = {"action": action.model_dump()}

        # Easy Task: Flag urgent emails
        if self.task_id == "easy":
            if action.action_type == ActionType.FLAG:
                # Check if email is urgent (contains "urgent" or "asap")
                is_urgent = "urgent" in email["subject"].lower() or "asap" in email["body"].lower() 
                if is_urgent:
                    reward_val = 1.0
                    info["correct"] = True 
                else: 
                    reward_val = -0.2  # penalty for false flag
                    info["correct"] = False 
            elif action.action_type == ActionType.ARCHIVE:
                # Archiving a non-urgent email is good; ignoring an urgent is bad
                is_urgent = "urgent" in email["subject"].lower() or  "asap" in email["body"].lower()
                if is_urgent:
                    reward_val = -0.5  # penalty for ignoring urgent
                else: 
                    reward_val = 0.1    # small reward for archiving non-urgent


            else: 
                # Any other action is invalid for this task
                reward_val = -0.1 
            
            # Reward the action for later grading 
            self.action_history.append({
                "email_id": email["id"],
                "action": action.action_type,
                "parameters": action.parameters,
                "subject": email["subject"],
                "body": email["body"]
            })
        
        elif self.task_id == "medium":
            # Medium: assign priority (high, medium, low)
            if action.action_type == ActionType.SET_PRIORITY:
                assigned = action.parameters.get("priority", "low")
                true_priority = email["true_priority"]
                if assigned == true_priority:
                    reward_val = 1.0
                elif (true_priority == "high" and assigned == "medium") or (true_priority == "medium" and assigned == "low"):
                    reward_val = 0.3 # partial credit for close guess
                else: 
                    reward_val = -0.2 
                info["assigned"] = assigned
                info["true"] = true_priority 
            else: 
                reward_val = -0.1 # penalty for wrong action type 

            self.action_history.append({
                "email_id": email["id"],
                "action": action.action_type,
                "parameters": action.parameters,
                "true_priority": email["true_priority"]
            })

        elif self.task_id == "hard":
            # Hard: replt with extracted meeting info
            if action.action_type == ActionType.REPLY:
                reply_text = action.parameters.get("reply_text", "")
                has_time = EXPECTED_EXTRACTION["time"].lower() in reply_text.lower()
                has_loc = EXPECTED_EXTRACTION["location"].lower() in reply_text.lower()

                if has_time and has_loc:
                    reward_val = 1.0
                elif has_time or has_loc:
                    reward_val = 0.5
                else: 
                    reward_val = 0.0
                info["reply"] = reply_text
                info["has_time"] = has_time
                info["has_location"] = has_loc
            else:
                reward_val = -0.2
            
            self.action_history.append({
                "email_id": email["id"],
                "action": action.action_type,
                "parameters": action.parameters,
                "subject": email["subject"],
                "body": email["body"]
            })

        # Move to next email
        self.current_idx += 1
        if self.current_idx >= len(self.emails):
            self.done = True 

        self.total_reward += reward_val
        obs = self._get_obs()
        obs.last_reward = reward_val 
        return obs, Reward(value=reward_val), self.done, info 

    def state(self) -> Dict:
        """Return the full internal state (useful for debugging)."""
        return {
            "task_id": self.task_id,
            "current_idx": self.current_idx,
            "done": self.done,
            "total_reward": self.total_reward, 
            "action_history": self.action_history
        }
# ========== Graders (0.0-10) ==========
# Simple Grader for Easy Task 
def grade_easy(env: EmailTriageEnv) -> float:
    """Compute F1 score for flagging urgent emails."""
    history = env.action_history 
    if not history:
        return 0.0
    true_pos = false_pos = false_neg = 0
    for item in history:
        # Determine if email was actually urgent
        is_urgent = "urgent" in item["subject"].lower() or "asap" in item["body"].lower()
        flagged = (item["action"] == ActionType.FLAG)
        if is_urgent and flagged:
            true_pos += 1
        elif not is_urgent and flagged:
            false_pos += 1
        elif is_urgent and not flagged:
            false_neg += 1
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0 
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall/ (precision + recall) if (precision + recall) > 0 else 0
    return f1 

# Grader for Medium Task
def grade_medium(env: EmailTriageEnv) -> float:
    """Accuracy for priority assignment."""
    history = env.action_history 
    if not history:
        return 0.0
    correct = 0
    for item in history:
        if item["action"] == ActionType.SET_PRIORITY:  
            assigned = item["parameters"].get("priority", "low") 
            if assigned == item["true_priority"]:
                correct += 1
    
    # Note: If agent never set_priority, correct stays 0 
    return correct / len(history)

# Grader for Hard Task
def grade_hard(env: EmailTriageEnv) -> float:
    """Check if the final reply contains both time and location."""
    history = env.action_history 
    if not history:
        return 0.0
    # Find the last REPLY action
    last_reply = None 
    for item in reversed(history):
        if item["action"] == ActionType.REPLY: 
            last_reply = item 
            break 
    if not last_reply: 
        return 0.0 
    reply_text = last_reply["parameters"].get("reply_text", "")
    has_time = EXPECTED_EXTRACTION["time"].lower() in reply_text.lower()
    has_loc = EXPECTED_EXTRACTION["location"].lower() in reply_text.lower() 

    if has_time and has_loc:
        return 1.0
    elif has_time or has_loc:
        return 0.5
    else:
        return 0.0
    
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard 
}
if __name__ == "__main__":
    # Quick test
    for task in ["easy", "medium", "hard"]:
        env = EmailTriageEnv("easy") 
        obs = env.reset()
        print(f"\n=== Testing {task} task ===")

        # Simulate a simple policy: flag if contains 'urgent' or 'asap'
        while not obs.done:
            # Heuristic policy:
            if task == "easy":
                # Flag if urgent 
                if "urgent" in obs.current_email_subject.lower() or "asap" in obs.current_email_body.lower():
                    action = Action(action_type=ActionType.FLAG)
                else: 
                    action = Action(action_type=ActionType.ARCHIVE)
            elif task == "medium":
                # Guess priority based on keywords 
                text = (obs.current_email_subject + " " + obs.current_email_body).lower()
                if any(k in text for k in ["urgent", "deadline", "critical", "failure", "asap"]):
                    priority = "high" 
                elif any(k in text for k in ["meeting", "reminder", "team"]):
                    priority = "medium"
                else: 
                    priority = "low"
                action = Action(action_type=ActionType.SET_PRIORITY, parameters={"priority": priority})
            else: # hard
                # Reply with extracted info if (assuming it's the last email) 
                
                action = Action(action_type=ActionType.REPLY, parameters=
                                {"reply_text": "Confirmed. Time: 2pm, Location: Blue conference room."})
                
                obs, reward, done, info = env.step(action) 
                print(f"Step: {obs.email_index}: {action.action_type.value} -> reward: {reward.value:.2f}")

            score = GRADERS[task](env)  
            print(f"Grader score for {task}: {score:.2f}")
            