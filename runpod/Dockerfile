FROM python:3.12-slim-bookworm
ARG SOURCE_SHA
ENV WAFERVLM_SOURCE_SHA=${SOURCE_SHA} \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache \
    PATH=/app/.venv/bin:$PATH
WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.5
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --extra train --no-dev
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY resume-input ./resume-input
RUN python -m unittest discover -s tests -v
ENTRYPOINT ["python", "scripts/run_experiment.py"]
