FROM python:3.12.13-slim

RUN groupadd -r router && useradd -r -g router -m router

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data && chown -R router:router /app

USER router
EXPOSE 8400

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8400"]
