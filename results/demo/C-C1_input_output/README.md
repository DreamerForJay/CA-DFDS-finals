# C-C1 輸入 → 輸出（近／中／遠）

- 紅色車 1.2 m（近）、3.7 m（中）、6.7 m（遠），shot01。每個距離 7 步，每步一張 1600×1200 PNG：`steps/<距離>/0_input.png … 6_result.png`，可直接做成網頁的「步驟滑桿」。
- `steps/<距離>/strip.png`（= `keyframes/strip_<距離>.png`）：7 步橫排一張，給簡報／列印用。
- `steps.json`：每步的數值（遮罩像素數、J／a_c／b_c 平均、各通道拉普拉斯標準差、權重、σ、校正係數、估測距離、誤差）。
  `sigma` 是由這次重算的拉普拉斯圖直接取標準差；`sigma_from_features` 是 `features/test_red.npz` 的值，兩者一致。
- 顯示說明：③–⑥ 為車輛區域放大；⑥ 的色階三個距離共用（以 6.7 m 的 99 百分位為上限、平方根色階），越亮越清晰。對焦在 15 m，所以車越近越模糊、σ 越小。
- 注意：論文中遮罩用 SAM 2（2024）；此處用資料集手繪遮罩（與實驗數字一致）。全自動遮罩（YOLO11 → SAM）見 GUI 與 RESULTS 補充 H。
- 影片：`video.mp4`（三個距離 × 7 步，31.3 s）、`video_short.mp4`（只有 3.7 m，11.0 s）。
- 產生：`python scripts/make_demo_assets.py --only C-C1`
