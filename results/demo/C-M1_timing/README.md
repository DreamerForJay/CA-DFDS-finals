# C-M1 即時性（處理時間）

- `timing.png`、`timing.json`：CA-DFDS 整條管線每張時間（讀圖 → 遮罩 → CIECAM02 → 拉普拉斯 → 校正）。
  GPU 版 CIECAM02（`cadfds_gpu.py`）與 CPU 結果相同（RMSE 一致），fp32 全圖約 73 ms／張（14.11 FPS）。
- 全自動遮罩 YOLO11 → SAM：yolo_detect 10.25 ms, sam 106.1 ms, wall_incl_io 198.31 ms（中位數）。
- 硬體：RTX 5080 + Intel Core Ultra 7 265K（實驗室主機）。量測腳本 `scripts/m3_timing_full_pipeline.py`（RESULTS 補充 K）。
- 畫面角落顯示 FPS 的錄影：GUI v3 的「計時與硬體」卡即時顯示（見 C-G1 GUI 錄影）。
