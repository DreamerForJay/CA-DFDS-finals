# 變更紀錄（CHANGELOG）

每次有更動請在**最上方**新增一筆：日期、執行者、做了什麼、影響的檔案、結果。規則見 [AGENTS.md](../AGENTS.md)。

---

## 2026-11-15｜上傳 Google Drive、修正 commit 作者、建立協作說明

執行者：Claude Code（實驗室 RTX 5080 主機），應 repo 擁有者要求

1. **修正 commit 作者**
   - 問題：實驗室主機的全域 Git 身分屬於他人，GitHub 將 2026-11-15 的 8 個 `lab results` commit 顯示為 `wiwihuang-cs` 所提交。
   - 處理：以 `git filter-branch --env-filter` 只改寫這 8 個 commit 的作者／提交者為 `DreamerForJay <DreamerForJay@users.noreply.github.com>`（內容、訊息、時間不變；改寫前後檔案樹已比對相同），並 `push --force-with-lease`。
   - 預防：在本 repo 設定 `git config --local user.name/user.email` 為 DreamerForJay；全域設定未更動。
   - 新舊 commit 對照：`1aeabbb→4e8948c`、`338546b→d27caa2`、`d1fe493→520c278`、`82487cf→4c0645a`、`743cd66→a4051a3`、`a211975→e34bb8f`、`279aa8e→65b1bbf`、`bbe0f65→98a297f`。
2. **上傳 Google Drive**（<https://drive.google.com/drive/folders/1L9zNHvzAKqQlTgFMuWbNuePYAKy4IThV>）
   - 工具：rclone v1.76.0（官方免安裝版，已驗證 SHA256），放在 `D:\Carla-0.10.0-Win64-Shipping\tools\rclone\`；remote `cadfds_gdrive` 以 `root_folder_id` 限定只能操作上述資料夾，由使用者於瀏覽器授權。
   - 上傳：`CA-DFDS-finals/`（repo 鏡像，不含 `.git`、`.venv-cadfds`、`logs/bin`、`logs/scratch`）與 `cadfds_results_20261115.zip`（成果壓縮檔）。
     - `CA-DFDS-finals/`：<https://drive.google.com/drive/folders/1dhRViJVTi19B6Qj7G2V88P_CmXReAIpD>（71 個檔案，`rclone check` 0 差異）
     - `cadfds_results_20261115.zip`（7.3 MB）：<https://drive.google.com/file/d/1yFIqqWjWMSxpD6vyn_aymB_9c4OBez7Q/view>
     - 內容：`results/`、`features/`、`logs/*.log|*.sh|*.py`、`RESULTS.md`、`AGENTS.md`、`docs/`
     - 分享權限沿用 Drive 父資料夾設定，未另外開放。
     - 過程中遇到 rclone 共用 client_id 被限流（403 Quota exceeded），以限速重試解決。
   - 新增 `scripts/sync_drive.sh` 供之後同步使用。
3. **協作說明**：新增 [AGENTS.md](../AGENTS.md)（環境、資料位置、Git 身分、Drive 同步、協作規則）、`CLAUDE.md`（指向 AGENTS.md）與本 CHANGELOG；README 開頭加上指引。

## 2026-11-15｜README 任務 A–G 全部完成

執行者：Claude Code（實驗室 RTX 5080 主機）

- 任務 A：抽取 28 組特徵（校正 6 色、測試 22 色）→ `features/*.npz`
- 任務 B：表 8 於測試距離 1.2–6.7 m 重做、式 (5)／(7) 重新最佳化 → `results/table8_testdist.csv`、`results/reopt_eq*.json`、`results/calibration_opt.json`
- 任務 C：demo 影片 → `results/CA-DFDS_demo_testdist.mp4`
- 任務 D：處理時間 → `results/timing.json`
- 任務 E：22 色 × 6 種設定 → `results/19colors_opt.csv`
- 任務 F：學長原始程式 → `results/original_code/`
- 任務 G：GUI 截圖 → `results/gui_*.png`
- 文件：`RESULTS.md`、`docs/IMPROVEMENT_PLAN.md`、`docs/LAB_REPORT_2026-11-15.md`
- 重要發現：校正用黑色車影像已於 2026-07-31 重新產生，論文校正曲線無法以現有資料重現（詳見報告第 4 節）
