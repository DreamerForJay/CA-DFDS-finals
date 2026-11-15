# AGENTS.md — 給協作 AI／同學的工作說明

本檔是所有 AI（Claude、Codex、Copilot…）與同學接手本 repo 時**第一個要讀的檔案**。
專案背景與任務清單見 [README.md](README.md)；最新進度見 [docs/CHANGELOG.md](docs/CHANGELOG.md)。

## 1. 目前狀態（2026-11-16）

- README 任務 A–G **全部完成**，數值結果見 [RESULTS.md](RESULTS.md)，完整流程與發現見 [docs/LAB_REPORT_2026-11-15.md](docs/LAB_REPORT_2026-11-15.md)。
- **待決事項**：校正用黑色車影像已於 2026-07-31 重新產生，論文校正曲線（R² 0.969）無法以現有資料重現（現為 0.942）。需由學長／指導教授決定以哪一版為準（報告第 4 節）。
- 改善計畫的剩餘項目見 [docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md)（⬜／🔄）。
- 整體架構、遮罩方式（目前是手繪遮罩，SAM 已整合進 GUI，YOLO 尚未使用）、與其他方法比較：見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
- GUI demo v3（2026-11-16）：點兩下 `run_gui.bat`。可同時比較 6 種方法，全自動遮罩（YOLO → SAM）、可調整的框、影片輸入，並偵測硬體。說明見 [docs/GUI_DEMO.md](docs/GUI_DEMO.md)。
- 方法比較（480 張，整體 RMSE）：CA-DFDS 0.337 m、灰階 0.362 m；UniDepth 原始 1.879 m、線性校正 **0.357 m**；Depth Pro 原始 0.871 m、線性校正 0.751 m。全自動遮罩下 CA-DFDS 為 0.310 m（RESULTS 補充 H、J）。
- **報告口徑**：公平比較時，CA-DFDS 與 UniDepth 準確度相當；CA-DFDS 近距離（≤ 3.7 m）較準，UniDepth 遠距離較準。不要寫「大幅勝過深度模型」。
- 成果已上傳 Google Drive（第 4 節）；commit 作者已修正為 DreamerForJay。

### 下一步候選（尚未開始）

**以 repo 擁有者整理的 [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md)（M1–M10）為優先順序**，完成後更新該表狀態。以下為補充：

1. 黑色校正資料決定後，重跑任務 B／E 並更新 `RESULTS.md`。
2. 改善計畫中的 ⬜ 項目，例如：YOLO → SAM 全自動遮罩（C3／C8）、用 SAM 遮罩重跑表 8、深度模型加線性校正的公平對照、以 GPU 加速 CIECAM02、找回近距離 GUI。
3. rclone 共用 client_id 停用前，改用自己的 client_id（第 4 節）。

## 2. 協作規則（必須遵守）

1. **每次有更動都要同步說明**：
   - 在 [docs/CHANGELOG.md](docs/CHANGELOG.md) 最上方新增一筆（日期、執行者、做了什麼、影響的檔案、結果）。
   - 有新數值結果 → 更新 `RESULTS.md`；完成改善項目 → 更新 `docs/IMPROVEMENT_PLAN.md` 狀態。
   - 流程、環境或路徑有變 → 更新本檔。
   - 然後 `git commit` + `git push`；若結果檔有變，再執行 `bash scripts/sync_drive.sh` 同步 Google Drive。
2. **Git 身分**：在本 repo 一律以 `DreamerForJay <DreamerForJay@users.noreply.github.com>` 提交。實驗室主機已在 repo 內設定（`git config --local`）；**不要使用或修改電腦的全域 Git 身分**——那是別人的帳號，曾造成 commit 掛到他人名下（已於 2026-11-15 修正）。提交前可用 `git config user.email` 確認。
3. **不要改寫已推送的歷史**（`push --force`、rebase 已推送的 commit），除非 repo 擁有者明確要求。
4. **commit 訊息**沿用 `lab results: <任務代號與一句摘要>` 或 `docs: …`／`scripts: …`。
5. **不要提交**：影像資料、`.venv-cadfds/`、`logs/bin/`（ffmpeg 執行檔）、`logs/scratch/`、任何憑證檔（如 `rclone.conf`）。單檔 > 95 MB 改放 Drive 並在 CHANGELOG 寫連結。
6. 結果檔放在 `results/`（需 `git add -f`）；特徵檔 `features/*.npz` 被 `.gitignore` 排除，也需 `git add -f`。

## 3. 實驗室主機環境（RTX 5080 主機）

| 項目 | 位置／內容 |
|---|---|
| repo | `D:\Carla-0.10.0-Win64-Shipping\CA-DFDS-finals` |
| Python 環境 | `.venv-cadfds`（Python 3.12.7、numpy 1.26.4、colour-science 0.4.6）；啟用：`.venv-cadfds\Scripts\activate`。既有的 `venv-carla` 裡的 `colour` 是**錯的同名套件**，不能用 |
| ffmpeg | `logs/bin/ffmpeg.exe`（取自 imageio-ffmpeg，未提交）；`make_demo_video.py` 前先把 `logs/bin` 加入 PATH |
| 資料 `<DATA>` | `D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas`（56 GB，不在 git 與 Drive） |
| 校正集 | `<DATA>/training/firetruck_<色>/Test_dataset_Focus1500cm_f1.4_f22.0`（1.0–7.0 m），遮罩 `<DATA>/training/Training_dataset_mask` |
| 測試集 | `<DATA>/firetruck_<色>/Test_dataset_Focus1500cm_f1.4_f22.0`（1.2–7.2 m；遮罩只有 T1–T12 → 實際 1.2–6.7 m） |
| 共用測試遮罩 | `<DATA>/firetruck_black/Test_dataset_Focus1500cm_f1.4_f22.0_mask`（各色遮罩已驗證完全相同） |
| 學長原始程式 | `Carla-0.10.0-Win64-Shipping/test_ciecam.ipynb`（已複製到 `results/original_code/`） |
| torch 環境 | `D:\Carla-0.10.0-Win64-Shipping\.venv`（torch 2.11 + cu128、ultralytics 8.4（YOLO11m）、segment_anything、UniDepth、Depth Pro；RTX 5080 可用）。GUI 會自動以子程序啟動 `baselines/deep_worker.py`。模型權重在 `Carla-0.10.0-Win64-Shipping/models*`；UniDepth 權重在 HuggingFace 快取，需設定 `HF_HUB_OFFLINE=1` |
| rclone | `D:\Carla-0.10.0-Win64-Shipping\tools\rclone\rclone.exe`，remote 名稱 `cadfds_gdrive`（設定檔在使用者 AppData，勿提交） |

注意：PowerShell 的萬用字元路徑（如 `T*\R`）可能不展開，統計檔案請用 Git Bash。

## 4. Google Drive 共用資料夾

- 位置：<https://drive.google.com/drive/folders/1L9zNHvzAKqQlTgFMuWbNuePYAKy4IThV>
- 內容配置：
  - `CA-DFDS-finals/`：本 repo 的鏡像（不含 `.git`、venv、ffmpeg）— <https://drive.google.com/drive/folders/1dhRViJVTi19B6Qj7G2V88P_CmXReAIpD>
  - `cadfds_results_<日期>.zip`：每次實驗之成果壓縮檔（README 第 3 節回傳清單）；2026-11-15 版：<https://drive.google.com/file/d/1yFIqqWjWMSxpD6vyn_aymB_9c4OBez7Q/view>
  - 其他檔案（論文、說明書、會議報告等）為使用者原有內容，**不要刪改**
- 同步：在 repo 根目錄執行 `bash scripts/sync_drive.sh`（先加 `--dry-run` 檢查）。只會改動 `CA-DFDS-finals/` 子資料夾與 zip。
- rclone 使用內建共用 client_id，偶爾回 `403 Quota exceeded`，腳本已限速重試；若持續失敗，等幾分鐘再試，或依 <https://rclone.org/drive/#making-your-own-client-id> 建立自己的 client_id（該共用 client_id 將於 2026 年內停用）。

## 5. 常用指令

```bash
# 重算特徵（已存在者跳過）
bash logs/run_taskA.sh
# 表 8／19 色評估（會改寫 calibration_opt.json 到 --out 的資料夾）
python evaluate.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz --test features/test_*.npz --out results/xxx.csv
# 方法比較：深度模型批次評估（torch 環境）→ 校正曲線 → 合併比較表
../.venv/Scripts/python.exe baselines/eval_deep.py --set test --sam --auto --out results/deep_baselines.csv
../.venv/Scripts/python.exe baselines/eval_deep.py --set calib --colors white gray black --out results/deep_calib.csv
python scripts/build_calibrations.py && python scripts/eval_auto_mask.py && python scripts/compare_methods.py
# 開啟 GUI demo（或直接點兩下 run_gui.bat）
.venv-cadfds/Scripts/python.exe gui_demo.py
# GUI 自動截圖
python logs/gui_screenshots.py results <影像> <遮罩> [...]
# 同步 Drive
bash scripts/sync_drive.sh
```
