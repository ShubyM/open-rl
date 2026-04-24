# Use a bare Debian image (uv will automatically download Python per the .python-version file)
FROM debian:bookworm-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .
COPY dev/tools/pyproject.toml ./dev/tools/pyproject.toml
COPY examples/pyproject.toml ./examples/pyproject.toml

# Configure uv for Docker and caching
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package open-rl-client --no-install-project

# Copy client source code
COPY examples ./examples

# Sync the project itself (to seal the venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package open-rl-client

# Run the python script natively inside uv
CMD ["uv", "run", "--no-sync", "--package", "open-rl-client", "python", "examples/rl/rlvr/rlvr.py", "--jobs", "10", "--steps", "30"]
