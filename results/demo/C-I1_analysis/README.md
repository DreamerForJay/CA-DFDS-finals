# C-I1 分析圖表

- `error_vs_distance.png`：各距離 RMSE（5 種方法）。近距離 CA-DFDS／灰階較準，遠距離 UniDepthV2（線性校正）較準。
- `error_by_color.png`：紅／綠／藍／黑各車色 RMSE。
- `error_19colors.png`：22 個車色資料夾（`results/19colors_opt.csv`），CA-DFDS vs 灰階；CA-DFDS 較佳 12/22。
- `charts.json`：上述三張圖的原始數據＋失敗案例摘要（`results/failure_cases/summary.csv`），給網頁做互動圖（例如 Chart.js）。
- 數字出處：`results/method_comparison.json`、`results/19colors_opt.csv`。白底圖，可直接放簡報。
