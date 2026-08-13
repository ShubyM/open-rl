#!/usr/bin/env bash
set -euo pipefail

run_root() {
  if [ "${EUID}" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

apt_update() {
  run_root env DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
}

apt_install() {
  run_root env DEBIAN_FRONTEND=noninteractive apt-get -y -qq install "$@"
}

ensure_apt_prereqs() {
  apt_update
  apt_install ca-certificates curl gnupg make
}

install_docker() {
  echo "Installing Docker..."
  ensure_apt_prereqs
  run_root install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | run_root gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
  run_root chmod a+r /etc/apt/keyrings/docker.gpg

  . /etc/os-release
  arch="$(dpkg --print-architecture)"
  echo "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" |
    run_root tee /etc/apt/sources.list.d/docker.list >/dev/null

  apt_update
  apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  run_root systemctl enable --now docker
}

install_nvidia_container_toolkit() {
  echo "Installing nvidia-container-toolkit..."
  ensure_apt_prereqs
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    run_root gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
    run_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  apt_update
  apt_install nvidia-container-toolkit
}

install_kind() {
  echo "Installing kind..."
  ensure_apt_prereqs
  tmp="$(mktemp)"
  curl -fsSL -o "$tmp" https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
  run_root install -m 0755 "$tmp" /usr/local/bin/kind
  rm -f "$tmp"
}

install_kubectl() {
  echo "Installing kubectl..."
  ensure_apt_prereqs
  stable="$(curl -Ls https://dl.k8s.io/release/stable.txt)"
  tmp="$(mktemp)"
  curl -fsSL -o "$tmp" "https://dl.k8s.io/release/${stable}/bin/linux/amd64/kubectl"
  run_root install -m 0755 "$tmp" /usr/local/bin/kubectl
  rm -f "$tmp"
}

install_helm() {
  echo "Installing Helm..."
  ensure_apt_prereqs
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
}

install_uv() {
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | run_root env UV_INSTALL_DIR=/usr/local/bin sh
}

if [ ! -r /etc/os-release ]; then
  echo "This bootstrap script requires Ubuntu."
  exit 1
fi

. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
  echo "This bootstrap script requires Ubuntu."
  exit 1
fi

echo "Checking NVIDIA driver..."
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "Install the NVIDIA driver (>= r550) first"
  exit 1
fi
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
echo "Detected NVIDIA driver ${driver_version}"

if command -v docker >/dev/null 2>&1; then
  echo "Docker already installed."
else
  install_docker
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia-container-toolkit already installed."
else
  install_nvidia_container_toolkit
fi

if command -v kind >/dev/null 2>&1; then
  echo "kind already installed."
else
  install_kind
fi

if command -v kubectl >/dev/null 2>&1; then
  echo "kubectl already installed."
else
  install_kubectl
fi

if command -v helm >/dev/null 2>&1; then
  echo "Helm already installed."
else
  install_helm
fi

if command -v uv >/dev/null 2>&1; then
  echo "uv already installed."
else
  install_uv
fi

current_user="${USER:-$(id -un)}"
if [ "${EUID}" -ne 0 ] && ! id -nG "${current_user}" | tr ' ' '\n' | grep -Fxq docker; then
  echo "Adding ${current_user} to the docker group..."
  run_root usermod -aG docker "${current_user}"
  echo "Log out and log back in for docker group membership to take effect."
fi

echo "Installed versions:"
docker --version
nvidia-ctk --version
kind --version
kubectl version --client=true
helm version --short

echo "Next steps:"
echo "  bash hack/kind/kind-up.sh   (or: make kind-up)"
echo "  make kind-images && make kind-deploy && make kind-status"
