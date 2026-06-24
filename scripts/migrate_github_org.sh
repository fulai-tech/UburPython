#!/usr/bin/env bash
# 将 dwqnidq/UburNode 转移至 fulai-tech/UburPython，并更新本地 origin。
set -euo pipefail

SOURCE_OWNER="dwqnidq"
SOURCE_REPO="UburNode"
TARGET_OWNER="fulai-tech"
TARGET_REPO="UburPython"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

log() { printf '%s\n' "$*"; }

ensure_gh() {
  command -v gh >/dev/null 2>&1 || {
    log "请先安装 GitHub CLI: brew install gh"
    exit 1
  }
}

ensure_gh_auth() {
  if gh auth status >/dev/null 2>&1; then
    return 0
  fi
  log "需要登录 GitHub（浏览器授权一次）"
  gh auth login -h github.com -p https -s repo,admin:org
}

repo_exists() {
  gh api "repos/$TARGET_OWNER/$TARGET_REPO" >/dev/null 2>&1
}

transfer_repo() {
  if repo_exists; then
    log "目标仓库已存在: $TARGET_OWNER/$TARGET_REPO，跳过转移"
    return 0
  fi
  log "正在转移 $SOURCE_OWNER/$SOURCE_REPO → $TARGET_OWNER/$TARGET_REPO ..."
  gh api \
    --method POST \
    "repos/$SOURCE_OWNER/$SOURCE_REPO/transfer" \
    -f "new_owner=$TARGET_OWNER" \
    -f "new_name=$TARGET_REPO"
  log "转移请求已提交（组织可能需在 GitHub 邮件/通知中确认）"
}

wait_for_target() {
  local i
  for i in $(seq 1 30); do
    if repo_exists; then
      log "目标仓库可访问: https://github.com/$TARGET_OWNER/$TARGET_REPO"
      return 0
    fi
    sleep 2
  done
  log "等待目标仓库超时，请稍后在 GitHub 确认转移是否完成"
  return 1
}

update_local_remote() {
  cd "$ROOT"
  local new_url="https://github.com/$TARGET_OWNER/$TARGET_REPO.git"
  git remote set-url origin "$new_url"
  log "本地 origin 已更新: $new_url"
  git remote -v
}

main() {
  ensure_gh
  ensure_gh_auth
  transfer_repo
  wait_for_target || true
  update_local_remote
  log ""
  log "=== 后续步骤 ==="
  log "1. 在 GitHub 组织 fulai-tech 确认仓库转移（若需审批）"
  log "2. 检查 Actions Secrets（SSH_*）是否仍有效"
  local deploy_dir="${DEPLOY_DIR:-/opt/uburnode}"
  log "3. 服务器 $deploy_dir 执行: git remote set-url origin https://github.com/$TARGET_OWNER/$TARGET_REPO.git"
  log "4. 提交并 push 本仓库中 deploy.yml / setup_github_secrets.sh 的 URL 更新"
}

main "$@"
