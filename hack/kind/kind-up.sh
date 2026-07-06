#!/usr/bin/env bash
set -euo pipefail

install_hint() {
  case "$1" in
    docker) echo "Missing docker. Install Docker Engine: https://docs.docker.com/engine/install/" ;;
    kind) echo "Missing kind. Install kind >= 0.30: https://kind.sigs.k8s.io/docs/user/quick-start/#installation" ;;
    kubectl) echo "Missing kubectl. Install kubectl: https://kubernetes.io/docs/tasks/tools/" ;;
    helm) echo "Missing helm. Install Helm 3: https://helm.sh/docs/intro/install/" ;;
    nvidia-smi) echo "Missing nvidia-smi. Install the NVIDIA Linux driver >= r550." ;;
    nvidia-ctk) echo "Missing nvidia-ctk. Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" ;;
    *) echo "Missing $1." ;;
  esac
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Checking required tools..."
missing_tools=0
for tool in docker kind kubectl helm nvidia-smi nvidia-ctk; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    install_hint "$tool"
    missing_tools=1
  fi
done
if [ "$missing_tools" -ne 0 ]; then
  exit 1
fi

echo "Checking docker default runtime..."
default_runtime="$(docker info --format '{{.DefaultRuntime}}')"
if [ "$default_runtime" != "nvidia" ]; then
  echo "Docker default runtime is '$default_runtime', expected 'nvidia'."
  echo "Fix: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker"
  read -r -p "Run this fix now? [y/N] " confirm
  if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
    sudo systemctl restart docker
  else
    exit 1
  fi
fi

echo "Checking nvidia-container-runtime config..."
config_file="/etc/nvidia-container-runtime/config.toml"
if ! grep -Eq '^[[:space:]]*accept-nvidia-visible-devices-as-volume-mounts[[:space:]]*=[[:space:]]*true[[:space:]]*$' "$config_file"; then
  echo "NVIDIA runtime config is missing accept-nvidia-visible-devices-as-volume-mounts = true."
  echo "Fix: sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true"
  read -r -p "Run this fix now? [y/N] " confirm
  if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true
  else
    exit 1
  fi
fi

echo "Ensuring kind cluster open-rl exists..."
if ! kind get clusters | grep -Fxq "open-rl"; then
  kind create cluster --config "$script_dir/kind-config.yaml"
fi

echo "Verifying GPU visibility inside the kind node..."
if ! docker exec open-rl-control-plane nvidia-smi -L; then
  echo "GPU is not visible in the kind node. Recheck steps 2/3, restart docker, and recreate existing clusters."
  exit 1
fi

echo "Installing NVIDIA DRA driver..."
helm upgrade --install dra-driver-nvidia-gpu \
  oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
  --version 0.4.1 --create-namespace --namespace dra-driver-nvidia-gpu \
  --set gpuResourcesEnabledOverride=true \
  --set nvidiaDriverRoot=/ \
  --set resources.computeDomains.enabled=false

echo "Waiting for gpu.nvidia.com DeviceClass..."
deadline=$((SECONDS + 120))
until kubectl get deviceclass gpu.nvidia.com >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "Timed out waiting for DeviceClass gpu.nvidia.com."
    exit 1
  fi
  sleep 2
done

echo "kind GPU cluster is ready."
echo "Next steps:"
echo "  make kind-images && make kind-deploy"
echo "  make kind-status"
