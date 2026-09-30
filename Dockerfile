FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
EXPOSE 8080
CMD ["uvicorn", "llm_eval.api:app", "--host", "0.0.0.0", "--port", "8080"]
