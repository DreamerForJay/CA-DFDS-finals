# 變更紀錄（CHANGELOG）

每次有更動請在**最上方**新增一筆：日期、執行者、做了什麼、影響的檔案、結果。規則見 [AGENTS.md](../AGENTS.md)。

---

## 2026-11-16｜GitHub issue #1–#8 實驗

執行者：Claude Code（實驗室 RTX 5080 主機），應 repo 擁有者要求依 issue 執行。結果見 RESULTS「2026-11-16 GitHub issue 實驗」（補充 K–Q）。

- **#1（M3）完整管線計時**
  - 新增 `cadfds_gpu.py`：PyTorch 版 CIECAM02、拉普拉斯、共變異；與 CPU 特徵相對誤差 1.8e-13。
  - CIECAM02 轉換 2283 → 17.8 ms（fp64）；整條管線 10.8–14.1 FPS；含 YOLO＋SAM 約 5 FPS。
  - 檔案：`results/timing_full_pipeline.json`、`scripts/m3_timing_full_pipeline.py`
- **#2 前期研究硬體**：論文、簡報、中文初稿都沒有記載，需向作者詢問。
- **#3（M1）式 (5)／(7) 對照**：互有勝負，建議統一寫式 (5)。檔案：`results/eq5_vs_eq7.csv`、`scripts/m1_eq5_vs_eq7.py`
- **#4（M2）消融 E**
  - 6 色校正，權重 (0.9786, 0.0214, 0)；14 個未參與校正車色 0.303 m，與 J only 0.301 m 相當。
  - 檔案：`results/ablation_E*.{csv,json}`、`scripts/m2_ablation_E.py`
- **#5（M4）全自動遮罩**
  - 校正集也改用自動遮罩（IoU 0.892），全自動 CA-DFDS 0.317 m。
  - 檔案：`results/auto_mask.csv`、`results/auto_mask_summary.json`、`results/auto_mask_examples/`、`features/cal_auto_*.npz`、`scripts/m4_auto_mask_{calib,report}.py`
- **#6（M5）gamma**
  - 現有資料沒有 gamma 混用（log-log 斜率 0.84–1.06）。
  - 全部轉為線性光後，灰階 0.332 → 0.251、CA-DFDS 0.337 → 0.276 m，需以 CARLA 重渲染確認。
  - 檔案：`results/gamma_check.csv`、`results/19colors_gamma_unified.csv`、`features/g1_*.npz`、`scripts/m5_gamma.py`
- **#8（M7、M8）其他車種與失敗案例**
  - `datas/audi.etron` 與 `datas/ford.mustang` 其實是同一台 SUV（nissan.patrol_2021）。
  - 套用消防車校正 RMSE 1.27 m（高估約 1.1 m）。
  - 7.2 m 為 0.48 m；翻轉不變；曝光 ±30% 時 0.41–0.62 m；紋理與誤差 ρ = −0.44。
  - 檔案：`results/multi_vehicle.csv`、`results/failure_cases/`、`scripts/m78_vehicle_failure.py`、`logs/m78.log`
- **#7 實體相機**：本機無法執行。
- 文件：更新 RESULTS、NEXT_EXPERIMENTS（M1–M5、M8 ✅，M7 🔄）、IMPROVEMENT_PLAN（A3、C2 ✅，A7 🔄）、AGENTS。
- 已用 gh（DreamerForJay 登入，安裝於 `C:\Program Files\GitHub CLI`）在 issue #1–#8 各貼一則結果摘要留言；#1、#3、#4、#5、#6 可關閉，#2、#7、#8 保持開啟。
- GitHub Contributors：main 上所有 commit 都已是 DreamerForJay。但改寫前的 8 個舊 commit 仍以未參照物件留在 GitHub，因此頁面側欄的 Contributors 快取仍可能顯示他人帳號；需 repo 擁有者聯絡 GitHub Support 清除。

## 2026-11-16｜GUI 改名：學長舊版 與 新版 v3

執行者：Claude Code（實驗室 RTX 5080 主機），應 repo 擁有者要求

| 原名 | 新名 | 說明 |
|---|---|---|
| `gui_demo.py`（v3） | `gui_new_v3.py` | **新版 v3**，視窗標題與頁首加上「新版 v3」 |
| `run_gui.bat` | `run_gui_new.bat` | 啟動新版 |
| （commit 782066e 的初版 `gui_demo.py`） | `gui_old_senior.py` ＋ 新增 `run_gui_old_senior.bat` | **學長舊版**（論文權重、單一方法），從 git 歷史還原 |
| `results/gui_1–3.png` | `results/gui_old_senior_1–3.png` | 舊版截圖 |
| `results/gui_v3_*.png` | `results/gui_new_v3_*.png` | 新版截圖 |

- 學長舊版只改了視窗標題，並讓校正檔路徑以程式所在資料夾為準，運算邏輯不變。
- 已同步更新 README、AGENTS、RESULTS（任務 G、補充 I）、GUI_DEMO、ARCHITECTURE、IMPROVEMENT_PLAN、LAB_REPORT_2026-11-15、`logs/gui_screenshots.py`、`baselines/deep_worker.py` 中的檔名。
- 本 CHANGELOG 較早的條目保留當時的檔名，不回頭修改。

## 2026-11-16｜深度模型線性校正、全自動遮罩（YOLO → SAM）、GUI v3

執行者：Claude Code（實驗室 RTX 5080 主機），應 repo 擁有者要求

- **深度模型線性校正（公平比較）**：
  - 新增 `results/deep_calib.csv`：在 CA-DFDS 校正集上跑 UniDepth、Depth Pro，共 390 張。
  - `scripts/build_calibrations.py` 擬合 `unidepth_lin`、`depthpro_lin`。
  - 結果：UniDepth 1.879 → **0.357 m**，Depth Pro 0.871 → 0.751 m。CA-DFDS 為 0.337 m。
- **全自動遮罩**：
  - torch 環境安裝 `ultralytics` 8.4.214，權重放在 `models/yolo11m.pt`。只新增套件，沒有更動 torch、numpy、OpenCV 等原有套件。
  - `deep_worker.py` 新增 `detect`、`auto`：YOLO → SAM，影片追蹤，近距離後備。
  - `eval_deep.py` 新增 `--set calib` 和 `--auto`；新增 `scripts/eval_auto_mask.py`。
  - 結果：IoU 0.906；CA-DFDS 在全自動遮罩下 RMSE 0.310 m。
- **GUI v3**（`gui_demo.py`）：
  - 重新排版：頁首（含硬體狀態）、左欄步驟卡片、中間影像、右欄結果卡片。
  - 可調整框：拖角點和邊調大小、拖框內移動、滾輪縮放，SAM 即時重新分割。
  - 遮罩新增「自動偵測」；輸入新增影片模式（播放，每 N 格處理一次）。
  - 只有主方法有完整說明，其他方法以滑鼠提示呈現。
  - 深度模型可切換原始輸出或線性校正。
  - 分項計時以兩欄顯示。
- 新增 `scripts/make_test_video.py`（測試影片）；`compare_methods.py` 改為同時輸出手繪和自動遮罩兩組結果。
- 文件：
  - ARCHITECTURE.md 改寫：遮罩來源、三個 CA-DFDS 版本的差別、全自動流程、公平比較與各距離分析。
  - 更新 GUI_DEMO.md（v3）、RESULTS 補充 J、AGENTS（報告口徑）、NEXT_EXPERIMENTS（M4 ✅）、IMPROVEMENT_PLAN（C3、C8 ✅，新增 C9）。
- 截圖：`results/gui_v3_1`–`5`，v2 截圖已刪除。

## 2026-11-16｜GUI v2（方法比較）、UniDepth／Depth Pro／SAM 比較、架構說明

執行者：Claude Code（實驗室 RTX 5080 主機），應 repo 擁有者要求

- **GUI v2**（`gui_demo.py`，`run_gui.bat` 同步更新）：
  - 修正字被裁切：支援高 DPI、右側面板自動換行、視窗可縮放。
  - 選影像：資料集瀏覽器（車色選單、距離滑桿、張數、←→↑↓ 快捷鍵），可拖放檔案。
  - 遮罩：自動載入對應的手繪遮罩；「指定遮罩檔…」直接開在遮罩資料夾。
  - 遮罩模式三選一：資料集遮罩、SAM 框選（GPU）、手動塗抹，並顯示和手繪遮罩的 IoU。
  - **6 種方法同時比較**：CA-DFDS 三種權重、灰階 DFD、UniDepth V2、ML-Depth-Pro。
    - 配色區分：暖色是本研究、冷色是他人方法、灰色是傳統基準。
    - 結果表、距離數線、影像標籤即時更新，★ 標示本張最佳。
  - 自動偵測 CPU、GPU（型號、VRAM、CUDA），即時顯示計時。
  - 截圖改用 PrintWindow，只擷取 GUI 視窗；之前的全螢幕擷取會拍到其他視窗，例如遠端軟體通知。
- **新增 `baselines/`**：
  - `deep_worker.py`：在 torch 環境以子程序執行 SAM、UniDepth、Depth Pro。
  - `eval_deep.py`：批次評估。
- **新增 `scripts/`**：`build_calibrations.py`（各方法的校正曲線）、`compare_methods.py`（整體比較表）。
- `cadfds_core.py` 新增 `crop_to_mask`：只算遮罩外框範圍，結果與整張相同（已驗證差異為 0），遠距離由 2.3 s 降到 0.6 s。原有函式未變。
- **新結果**（RESULTS 補充 H、I）：
  - 480 張整體 RMSE：CA-DFDS 0.337、重新最佳化 0.314、J only 0.313、灰階 0.362、Depth Pro 0.871、UniDepth 1.879 m。
  - SAM 框選與手繪遮罩的 IoU 平均 0.945。
  - 新檔案：`results/method_comparison.*`、`deep_baselines.csv`、`calibration_methods.json`、`gui_v2_*.png`。
- **新增 [docs/ARCHITECTURE.md](ARCHITECTURE.md)**：整體流程、遮罩說明（目前用手繪遮罩，SAM 和 YOLO 的差異）、方法比較與公平性說明、硬體。
- NEXT_EXPERIMENTS.md：M3、M4 改為 🔄 並註明已量測與尚缺項目；新增「與單張深度模型比較」✅。
- 更新：GUI_DEMO.md（v2 操作）、AGENTS.md、README、IMPROVEMENT_PLAN（C3 🔄、C6、新增 C7 ✅、C8 ⬜）、`logs/gui_screenshots.py`（改呼叫 `--screenshot`）。

## 2026-11-15｜GUI demo 啟動方式與進度紀錄

執行者：Claude Code（實驗室 RTX 5080 主機），應 repo 擁有者要求

- 新增 `run_gui.bat`：點兩下即可用 `.venv-cadfds` 啟動 `gui_demo.py`，並自動載入 `results/calibration_opt.json`。
- 新增 [docs/GUI_DEMO.md](GUI_DEMO.md)：啟動方式、操作步驟、範例影像與遮罩路徑、在別台電腦執行的方法。
- 更新 `AGENTS.md`：第 1 節加入目前進度與下一步候選，第 5 節加入 GUI 指令；README 加入 GUI 說明連結。
- 程式與數值結果皆未變更。

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
