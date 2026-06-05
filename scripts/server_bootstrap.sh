#!/usr/bin/env bash
# 服务器首次引导：幂等安装 Docker Engine + Compose 插件（已安装则秒退）。
# 由 Deploy workflow 在每次部署前调用；重复执行不会重装。
set -euo pipefail

MARKER_FILE="${UBURNODE_BOOTSTRAP_MARKER:-/opt/uburnode/.docker_bootstrapped}"

log() { printf '[uburnode-bootstrap] %s\n' "$*"; }

docker_ready() {
  command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1
}

install_docker() {
  if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    log "检测到系统: ${NAME:-unknown} ${VERSION_ID:-}"
  fi
  log "正在安装 Docker（官方脚本）..."
  curl -fsSL https://get.docker.com | sh
}

enable_and_start() {
  if command -v systemctl >/dev/null 2>&1; then
    systemctl enable docker
    systemctl start docker
  fi
}

main() {
  if docker_ready; then
    log "Docker 已就绪，跳过安装"
    touch "$MARKER_FILE" 2>/dev/null || true
    docker compose version
    exit 0
  fi

  install_docker
  enable_and_start

  if ! docker_ready; then
    log "Docker 安装后仍不可用，请检查权限或服务状态"
    exit 1
  fi

  touch "$MARKER_FILE" 2>/dev/null || true
  log "Docker 安装完成"
  docker compose version
}

main "$@"
