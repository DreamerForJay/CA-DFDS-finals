# C-A1 Teaser（三聯畫面）

- 內容：紅色消防車由 6.7 m 接近到 1.2 m（資料集 T12→T01，每個距離 shot01）。
  左：輸入影像＋車輛遮罩＋估測距離；中：CIECAM02 加權拉普拉斯清晰度熱圖（車輛區域放大，所有幀共用同一色階）；右：估測 vs 真實距離曲線。
- 資料：CARLA 模擬，40 mm、f/1.4、對焦 15 m。遮罩為資料集手繪遮罩。**12 個距離是分別拍攝的靜態影像，不是連續影片**。
- 方法：CA-DFDS 論文權重 (0.8136, 0.1782, 0.0081)，校正曲線 `results/calibration_methods.json` 的 `opt`（白＋灰＋黑校正集）。
- 數字：本段 RMSE 0.295 m；片尾的 480 張 RMSE 0.337 m 出自 `results/method_comparison.json`。逐幀數值：`per_frame.csv`。
- 檔案：`video.mp4`（20.0 s）、`video_short.mp4`（13.2 s）、`poster.jpg`、`keyframes/`。
- 產生：`python scripts/make_demo_assets.py --only C-A1`
