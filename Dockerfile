FROM python:3.11-bookworm

ARG PYTORCH_INDEX_URL=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl

WORKDIR /workspace

COPY tokamak-sim/pyproject.toml tokamak-sim/README.md ./tokamak-sim/
COPY tokamak-sim/tokamak_control ./tokamak-sim/tokamak_control

COPY tokamak-rl/pyproject.toml tokamak-rl/README.md ./tokamak-rl/
COPY tokamak-rl/tokamak_rl ./tokamak-rl/tokamak_rl
COPY tokamak-rl/scripts ./tokamak-rl/scripts
COPY tokamak-rl/configs ./tokamak-rl/configs

RUN python -m pip install --upgrade pip \
    && python -m pip install ./tokamak-sim \
    && if [ -n "$PYTORCH_INDEX_URL" ]; then \
        python -m pip install --extra-index-url "$PYTORCH_INDEX_URL" './tokamak-rl[train]'; \
    else \
        python -m pip install './tokamak-rl[train]'; \
    fi

COPY tokamak-rl/tests ./tokamak-rl/tests

RUN mkdir -p /workspace/tokamak-rl/outputs \
    /workspace/tokamak-rl/checkpoints \
    /workspace/tokamak-rl/runs \
    /tmp/matplotlib

WORKDIR /workspace/tokamak-rl

CMD ["python", "scripts/train.py", "--help"]
