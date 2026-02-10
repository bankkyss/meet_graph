FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY meeting_minutes_graphrag_fastapi.py /app/meeting_minutes_graphrag_fastapi.py
COPY api /app/api
COPY services /app/services
COPY static /app/static

EXPOSE 8011

CMD ["uvicorn", "meeting_minutes_graphrag_fastapi:app", "--host", "0.0.0.0", "--port", "8011"]
