#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
  printf "\n[%s] %s\n" "$(date +'%H:%M:%S')" "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script currently supports Linux only."
  exit 1
fi

log "Installing system dependencies (ffmpeg, git-lfs)"
if require_cmd apt-get; then
  if require_cmd sudo; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg git-lfs
  else
    apt-get update
    apt-get install -y ffmpeg git-lfs
  fi
else
  echo "apt-get not found. Please install ffmpeg and git-lfs manually."
  exit 1
fi

log "Installing uv (if missing)"
if ! require_cmd uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

log "Creating virtual environment and installing dependencies"
cd "$PROJECT_ROOT"
uv sync

log "Initializing git lfs"
git lfs install

if [[ "${SKIP_WEIGHTS:-}" != "1" ]]; then
  if [[ ! -d "${PROJECT_ROOT}/pretrained_weights" ]]; then
    log "Cloning pretrained weights into pretrained_weights"
    git clone https://huggingface.co/lixinyizju/moda pretrained_weights
  else
    log "pretrained_weights already exists, skipping clone"
  fi
else
  log "Skipping pretrained weights download (SKIP_WEIGHTS=1)"
fi

log "Setup complete"
echo "Activate the environment with: source .venv/bin/activate"
