.PHONY: help push-vm pull-vm cluster-eval build-images push-images deploy-images push-to-cluster deploy deploy-fft-timeslice

# ---------------------------------------------------------------------------
# Deployment knobs
# ---------------------------------------------------------------------------
EVAL_MODEL_PATH ?=
EVAL_EXAMPLES ?= 100
EVAL_DATA_PATH ?=
EVAL_NAMESPACE ?=

help:
	@echo "Python setup: uv sync --extra cpu --extra cluster"
	@echo "Tests: uv run --no-sync pytest"
	@echo "Cluster job: uv run --no-sync openrl launch <recipe.py> --image <image>"
	@echo "Fast source deploy: uv run --no-sync openrl deploy"
	@echo "Slow image deploy: make push-to-cluster GCP_PROJECT=<project>"
	@echo "make cluster-eval EVAL_MODEL_PATH=/mnt/shared/open-rl/checkpoints/...  # one-off vLLM eval job on the cluster"

# ---------------------------------------------------------------------------
# Deployment (GKE)
# ---------------------------------------------------------------------------
GCP_PROJECT ?= cdrollouts-sunilarora
IMAGE_TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || cat VERSION 2>/dev/null || echo latest)
K8S_DIR     ?= k8s/deploy/distributed-fft-timeslice

build-images:
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) -f src/server/Dockerfile .
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG) -f src/server/Dockerfile.gateway .
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-client:$(IMAGE_TAG) -f src/server/Dockerfile.client .

push-images:
	docker push gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG)
	docker push gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG)
	docker push gcr.io/$(GCP_PROJECT)/open-rl-client:$(IMAGE_TAG)

deploy-images:
	kubectl apply -k $(K8S_DIR)
	kubectl set image deployment/open-rl-gateway gateway=gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG) 2>/dev/null || true
	kubectl set image daemonset/open-rl-accel-timeslicer accel-timeslicer=gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) 2>/dev/null || true
	kubectl set env deployment/open-rl-gateway OPEN_RL_WORKER_IMAGE=gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) OPEN_RL_WORKER_REVISION=$(IMAGE_TAG) 2>/dev/null || true
	kubectl rollout status deployment/open-rl-gateway --timeout=300s

# Slow path for dependency, runtime-image, and Kubernetes manifest changes.
# Ordinary Python edits can use `uv run --no-sync openrl deploy`.
push-to-cluster: build-images push-images deploy-images

deploy:
	kubectl apply -k k8s/deploy/distributed-lustre/

# FFT DRA variant: the gateway launches one worker pod per FFT model, all pinned
# to one physical GPU allocation via a shared DRA ResourceClaim.
# See docs/setup/gke-fft-timeslice.md.
deploy-fft-timeslice:
	kubectl apply -k k8s/deploy/distributed-fft-timeslice/

rollout:
	kubectl rollout restart deployment redis-store open-rl-gateway open-rl-trainer-worker vllm-worker

# One-off vLLM eval of a checkpoint on the shared PVC:
cluster-eval:
	@if [ -z "$(EVAL_MODEL_PATH)" ]; then \
	  echo "Missing EVAL_MODEL_PATH. Example:"; \
	  echo "  make cluster-eval EVAL_MODEL_PATH=/mnt/shared/open-rl/checkpoints/<model-id>/weights/final"; \
	  exit 2; \
	fi; \
	set -- --model-path "$(EVAL_MODEL_PATH)" --examples "$(EVAL_EXAMPLES)"; \
	if [ -n "$(EVAL_DATA_PATH)" ]; then set -- "$$@" --data-path "$(EVAL_DATA_PATH)"; fi; \
	if [ -n "$(EVAL_NAMESPACE)" ]; then set -- "$$@" --namespace "$(EVAL_NAMESPACE)"; fi; \
	python3 scripts/run_cluster_eval.py "$$@"

# Local Redis (for testing distributed mode):
#   sudo apt install redis-server && sudo service redis-server start
#   redis-cli ping   # should print PONG
#   sudo service redis-server stop

# GKE client jobs — run directly:
#   kubectl apply -f examples/rl/rlvr/rlvr-job.yaml
#   kubectl apply -f examples/rl/tinker-rl-basic/tinker-rl-basic-job.yaml
#   kubectl logs -f job/<job-name>
#   kubectl delete job <job-name>

dashboard-apply:
	@dev/monitoring/apply_dashboard.sh $(GCP_PROJECT)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
# Remote host address for VM synchronization. Override on command line: make push-vm REMOTE_HOST=...
REMOTE_HOST ?= <PLACE_HOLDER_FOR_REMOTE_HOST_ADDRESS>

# Push local workspace changes to the remote VM
push-vm:
	@git rev-parse --short HEAD > VERSION 2>/dev/null || true
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'scratch' ./ $(REMOTE_HOST):~/open-rl

# Pull changes from the remote VM back to the local workspace
pull-vm:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'scratch' $(REMOTE_HOST):~/open-rl/ ./
