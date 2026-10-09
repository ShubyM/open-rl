# syntax=docker/dockerfile:1
# OpenRL's worker image: vLLM's published image with OpenRL on top.
#
# The base already carries vLLM, torch, FlashInfer (with its JIT kernel cache),
# tilelang, nvcc and the Python headers runtime kernel builds need. This file
# only adds what OpenRL's samplers and trainers need beyond that, and never
# replaces a package the base installed. OpenRL hooks into vLLM through its
# vllm.general_plugins entry point; nothing in vLLM is patched.
#
# The CUDA 13 build: v0.30.0-cu129 ships a CUDA 13 torch 2.14 beside cu129
# torchvision, which cannot load. CUDA 13 needs node driver 580 or newer.
FROM vllm/vllm-openai:v0.30.0

# The base starts `vllm serve`; workers name their own command.
ENTRYPOINT []

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# git fetches the time-slicer client, a git dependency.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# The CUDA checkpoint/restore utility the snapshot agent drives. It is
# driver-backed; keep the node driver compatible.
RUN curl -fsSL https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint \
    -o /usr/local/bin/cuda-checkpoint \
    && chmod +x /usr/local/bin/cuda-checkpoint

WORKDIR /app
COPY pyproject.toml README.md ./

# Install OpenRL's own requirements that the base lacks, holding every package
# the base installed at its version, so nothing can swap vLLM's torch or
# FlashInfer and a real conflict fails the build. Only direct requirements are
# listed; their dependencies resolve against the base, so torch's are already
# met. They take versions that fit the base rather than our lock's, which
# describes another CUDA build. datasets is left out: no release accepts the
# base's fsspec, and no worker imports it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip freeze --system > /tmp/base.txt \
    && python3 - <<'EOF'
import re, tomllib
project = tomllib.load(open("pyproject.toml", "rb"))["project"]
wanted = project["dependencies"] + project["optional-dependencies"]["gpu"] + project["optional-dependencies"]["cluster"]
name = lambda req: re.split(r"[ =<>!~;@\[]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")
base = {name(line) for line in open("/tmp/base.txt") if line.strip() and not line.startswith("#")}
missing = [req for req in wanted if name(req) not in base and name(req) != "datasets"]
open("/tmp/missing.txt", "w").write("\n".join(missing) + "\n")
print(f"installing {len(missing)} of {len(wanted)} requirements: {', '.join(name(r) for r in missing)}")
EOF
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --constraint /tmp/base.txt -r /tmp/missing.txt

# OpenRL itself. Installing it registers the vLLM plugin entry point.
COPY . .
RUN uv pip install --system --no-deps .

ENV CUDA_HOME="/usr/local/cuda" \
    # GKE's device plugin mounts the NVIDIA driver (libcuda.so) at
    # /usr/local/nvidia; without it on the path torch and vLLM silently run on CPU.
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:$LD_LIBRARY_PATH"

CMD ["python3", "-m", "uvicorn", "server.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
