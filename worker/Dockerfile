FROM python:3.11-slim

WORKDIR /app

COPY worker/worker.py .
COPY prompts/teacher_prompt.txt .

ENV PYTHONUNBUFFERED=1
ENV PROMPT_PATH=/app/teacher_prompt.txt

CMD ["python", "-u", "worker.py"]