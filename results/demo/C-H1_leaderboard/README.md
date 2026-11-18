# C-H1 排行榜表（CARLA 480 張）

- `leaderboard.csv`／`leaderboard.json`（網頁可排序表格用）、`leaderboard.md`（簡報、README 用）。
- 欄位：整體 RMSE／MAE／偏差、各車色 RMSE、近距離（1.2–3.7 m）與遠距離（4.2–6.7 m）RMSE、全自動遮罩（YOLO11 → SAM）RMSE、旋轉 180° 後 RMSE、每張時間。
- 年份與出處：STMAP = Subbarao & Surya, “Depth from defocus: A spatial domain approach,” IJCV 13(3), 1994；
  Depth Pro = Bochkovskii et al., arXiv 2410.02073（ICLR 2025）；UniDepthV2 = Piccinelli et al., arXiv 2502.20110（2025）。
- 時間口徑不同：CA-DFDS／灰階／STMAP 為 CPU 單執行緒、含讀圖（`results/timing.json`）；深度模型為 RTX 5080 只計推論。CA-DFDS GPU 版見 C-M1。
- 20–110 cm 實拍結果（前期研究 DFDS 論文 Table 3/4）不在本表；本表全部是本機重新量測的 CARLA 結果。
- 數字出處：`results/method_comparison.json`（`scripts/compare_methods.py`）、`../C-F1_before_after/rotation_summary.json`。
