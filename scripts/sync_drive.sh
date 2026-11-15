#!/bin/bash
# 將本 repo 與成果壓縮檔同步到 Google Drive 共用資料夾。
# Drive 資料夾：https://drive.google.com/drive/folders/1L9zNHvzAKqQlTgFMuWbNuePYAKy4IThV
# rclone remote「cadfds_gdrive」的根目錄已設為上述資料夾（root_folder_id），只能操作此資料夾內容。
#
# 用法（在 repo 根目錄，Git Bash）：
#   bash scripts/sync_drive.sh                 # 同步專案資料夾 + 上傳 ../cadfds_results_*.zip
#   bash scripts/sync_drive.sh --dry-run       # 只列出會變更的檔案
# 前置：實驗室主機已有 D:\Carla-0.10.0-Win64-Shipping\tools\rclone\rclone.exe 與已授權之 remote。
# 其他電腦：安裝 rclone 後執行
#   rclone config create cadfds_gdrive drive scope=drive root_folder_id=1L9zNHvzAKqQlTgFMuWbNuePYAKy4IThV
set -euo pipefail
cd "$(dirname "$0")/.."
RCLONE=${RCLONE:-/d/Carla-0.10.0-Win64-Shipping/tools/rclone/rclone.exe}
command -v "$RCLONE" >/dev/null 2>&1 || RCLONE=rclone
REMOTE=cadfds_gdrive:
# rclone 內建共用 client_id 常被 Google 限流（403 Quota exceeded），因此限速並重試
RATE=(--tpslimit 2 --retries 5 --low-level-retries 10)

# 1) 專案資料夾 → Drive:/CA-DFDS-finals（鏡像；只影響此子資料夾，其中多出的檔案會被刪除）
"$RCLONE" sync . "${REMOTE}CA-DFDS-finals" "${RATE[@]}" "$@" \
  --exclude ".git/**" --exclude ".venv-cadfds/**" --exclude "logs/bin/**" --exclude "logs/scratch/**" \
  --exclude "__pycache__/**" --stats-one-line -v

# 2) 成果壓縮檔 → Drive 根目錄（只新增／更新，不刪除舊版）
for z in ../cadfds_results_*.zip; do
  [ -f "$z" ] && "$RCLONE" copy "$z" "$REMOTE" "${RATE[@]}" "$@" --stats-one-line -v
done

"$RCLONE" lsf "$REMOTE" --max-depth 1 "${RATE[@]}"
