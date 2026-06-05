FROM python:3.11-slim-bookworm

ENV APP_HOME=/app
ENV PYTHONUNBUFFERED=True
ENV PORT=8050
ENV WMATA_DATA_MODE=sample

WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn --bind 0.0.0.0:${PORT} --log-level info --workers 1 --threads 4 --timeout 0 Dashboard:server
