FROM python:3.10-slim 

WORKDIR /app 

# Copy requirements first for better caching 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the application 
COPY . .

# Expose the port that Gradio runs on
EXPOSE 7860 

# Start FastAPI server in background and Gradio in foreground
CMD ["sh", "-c", "python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 & python app.py"]
