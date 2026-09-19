FROM python:3.12-slim

WORKDIR /app

COPY api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY api/api.py ./api/api.py
COPY files /app/files
COPY results /app/results
COPY models /app/models

EXPOSE 8000

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
