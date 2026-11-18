# C-E1 四方法同步比較（1×4）

- 同一張影像、同一幀，四格：Depth Pro（2024）、UniDepthV2（2025）、STMAP（1994）、CA-DFDS（本作品，金框）。每格左下角為該方法估測距離與誤差；下方為逐幀絕對誤差曲線。
- 影片 `video.mp4`：紅→綠→藍→黑，每色 6.7 m → 1.2 m 共 12 個距離（shot01），52.07 s；`video_short.mp4` 只有紅色車，13.5 s。片尾為 480 張整體 RMSE（`results/method_comparison.json`）。
- `panels/<距離>_<方法>.jpg`：紅色車 12 個距離 × 4 方法單格大圖（1600×1200），對應 DEMO_PLAN D4。
- `per_frame.csv`：48 幀 × 各方法估測、誤差，以及深度模型原始輸出（網頁畫曲線用）。`summary.json`：這 48 幀與 480 張的 RMSE。
- 各方法設定：
  - **CA-DFDS**：單張 f/1.4 影像＋遮罩，論文權重，校正 `opt`。格內為清晰度熱圖。
  - **STMAP**（Subbarao & Surya 1994，空間域 DFD）：需要 f/1.4 與 f/22 **兩張影像**；沒有分割能力，在遮罩外框內整塊計算（`scripts/stmap_eval.py`，與學長 `stmap_dfds_rotation_eval.py` 同公式），二次校正曲線（同一校正集）。格內綠框為取樣框。
  - **UniDepthV2 / Depth Pro**：官方權重、未重新訓練，相機焦距 1137.8 px，取遮罩內深度中位數，再以同一校正集做線性校正 d = a·pred + b。格內為深度圖（0–10 m，紅 = 近），白線為遮罩。原始輸出也列在 CSV。
- 480 張整體 RMSE：CA-DFDS 0.337 m、UniDepthV2 線性校正 0.357 m（原始 1.879）、STMAP 0.478 m、Depth Pro 線性校正 0.751 m（原始 0.871）。
- 解讀口徑（AGENTS.md）：公平比較時 CA-DFDS 與 UniDepthV2 相當；近距離 CA-DFDS 較準、遠距離 UniDepthV2 較準。不要寫「大幅勝過深度模型」。
- 產生：`../.venv/Scripts/python.exe baselines/demo_deep.py`（深度圖與旋轉推論）→ `python scripts/make_demo_assets.py --only C-E1`
