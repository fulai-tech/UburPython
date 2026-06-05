#!/usr/bin/env bash
# 一键：生成部署 SSH 密钥 → 写入 GitHub Actions Secrets（需本机已 gh auth login）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${GITHUB_REPO:-dwqnidq/UburNode}"
KEY_PATH="${DEPLOY_KEY_PATH:-$HOME/.ssh/uburnode_deploy}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/uburnode}"

log() { printf '%s\n' "$*"; }

ensure_gh() {
  if command -v gh >/dev/null 2>&1; then
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    log "正在安装 GitHub CLI (gh)..."
    brew install gh
    return 0
  fi
  log "未找到 gh。请安装: https://cli.github.com/ 或 macOS: brew install gh"
  exit 1
}

ensure_gh_auth() {
  if gh auth status >/dev/null 2>&1; then
    return 0
  fi
  log "需要登录 GitHub（浏览器授权一次即可）"
  gh auth login -h github.com -p https -s repo,write:packages,read:packages
}

ensure_ssh_key() {
  if [[ -f "$KEY_PATH" && -f "${KEY_PATH}.pub" ]]; then
    log "已存在部署密钥: $KEY_PATH"
    return 0
  fi
  log "生成部署专用 SSH 密钥: $KEY_PATH"
  ssh-keygen -t ed25519 -C "github-actions-uburnode" -f "$KEY_PATH" -N ""
}

main() {
  ensure_gh
  ensure_gh_auth
  ensure_ssh_key

  log ""
  log "=== 公钥（需写入服务器 ~/.ssh/authorized_keys）==="
  cat "${KEY_PATH}.pub"
  log ""

  if [[ -z "${SSH_HOST:-}" ]]; then
    read -rp "服务器公网 IP (SSH_HOST): " SSH_HOST
  fi
  SSH_USER="${SSH_USER:-root}"
  read -rp "SSH 用户名 [${SSH_USER}]: " input_user
  SSH_USER="${input_user:-$SSH_USER}"

  SSH_PORT="${SSH_PORT:-22}"
  read -rp "SSH 端口 [${SSH_PORT}]: " input_port
  SSH_PORT="${input_port:-$SSH_PORT}"

  log ""
  log "尝试把公钥复制到服务器（需输入服务器密码一次）..."
  if ssh-copy-id -i "${KEY_PATH}.pub" -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}"; then
    log "公钥已写入服务器"
  else
    log "ssh-copy-id 失败：请手动把上面公钥追加到服务器 authorized_keys"
    read -rp "完成后按回车继续..."
  fi

  if ! ssh -i "$KEY_PATH" -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 \
    "${SSH_USER}@${SSH_HOST}" "echo ssh_ok"; then
    log "无法用部署密钥 SSH 登录，请检查 authorized_keys 后重试"
    exit 1
  fi
  log "SSH 密钥登录验证通过"

  log "写入 GitHub Secrets (repo: $REPO)..."
  gh secret set SSH_PRIVATE_KEY <"$KEY_PATH" --repo "$REPO"
  gh secret set SSH_HOST --body "$SSH_HOST" --repo "$REPO"
  gh secret set SSH_USER --body "$SSH_USER" --repo "$REPO"
  gh secret set SSH_PORT --body "$SSH_PORT" --repo "$REPO"

  read -rp "仓库或 GHCR 镜像是否为私有？(y/N): " need_ghcr
  if [[ "${need_ghcr,,}" == "y" ]]; then
    log "在 GitHub → Settings → Developer settings 创建 classic token，勾选 read:packages"
    read -rsp "粘贴 GHCR_TOKEN: " ghcr_token
    echo
    gh secret set GHCR_TOKEN --body "$ghcr_token" --repo "$REPO"
  fi

  log ""
  log "=== 请在服务器创建 $DEPLOY_DIR/.env ==="
  log "  mkdir -p $DEPLOY_DIR && cp 项目内 .env.example 为 .env 后编辑"
  log "  ES_NODE 在 compose 中已覆盖为 http://elasticsearch:9200"
  log "  无需手动安装 Docker：首次 Deploy 会自动执行 server_bootstrap.sh（已安装则跳过）"
  log ""
  log "Secrets 已配置。在 Actions 里 Run workflow「Deploy UburNode」即可一键部署。"
}

main "$@"
