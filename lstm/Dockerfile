FROM ghcr.io/astral-sh/uv:0.9.10-python3.13-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

RUN mkdir -p artifacts plots

COPY __init__.py adaround.py apot.py efficientqat.py lsq.py lstm.py main_train.py pact.py plot_qat_results.py ./

VOLUME ["/app/artifacts", "/app/plots"]

CMD ["bash", "-lc", "uv run python main_train.py && uv run python plot_qat_results.py"]
