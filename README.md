---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Email Triage Environment

Real-world email triage environment for training AI agents with three difficulty levels.

## Tasks

- **Easy**: Flag urgent emails (identify emails with urgency keywords)
- **Medium**: Assign priority levels (high/medium/low) to emails
- **Hard**: Extract meeting information from conversation threads

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment:
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN or OPENAI_API_KEY
```

3. Run server:
```bash
python -m uvicorn server.app:app --port 7860
```

4. In another terminal, run inference:
```bash
python inference.py
```

## Docker

Build and run with Docker:
```bash
docker build -t email-triage .
docker run -p 7860:7860 --env-file .env email-triage
```

## API Endpoints

- `GET/POST /reset?task_id=easy` - Reset environment
- `POST /step` - Take an action
- `GET /state` - Get current state
- `GET /score_task?task_id=easy` - Get grader score
- `GET /tasks` - List available tasks

## Baseline Scores

- Easy: 0.667
- Medium: 0.875
- Hard: 1.0

## License

MIT
