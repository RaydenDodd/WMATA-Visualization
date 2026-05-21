FROM python:3.8-slim-buster

ENV APP_HOME /app
ENV PYTHONUNBUFFERED True
WORKDIR $APP_HOME

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir gunicorn

RUN groupadd -r app && useradd -r -g app app
COPY --chown=app:app . ./
USER app

CMD exec gunicorn --bind :$PORT --log-level info --workers 1 --threads 8 --timeout 0 app:server
