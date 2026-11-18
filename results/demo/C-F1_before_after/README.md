# C-F1 前後對照（旋轉 180°；灰階 vs CIECAM02）

**① 旋轉 180°**（論文 Table 4 的設定；測試集 480 張）

| 方法 | 原始 RMSE | 旋轉後 RMSE | 變化 |
|---|---|---|---|
| CA-DFDS（本作品） | 0.337 m | 0.337 m | +0.000 m |
| STMAP（1994） | 0.478 m | 0.785 m | +0.306 m |
| UniDepthV2（2025） | 0.357 m | 3.013 m | +2.655 m |
| Depth Pro（2024） | 0.751 m | 1.524 m | +0.773 m |
| UniDepthV2 原始輸出（未校正） | 1.879 m | 1.608 m | -0.271 m |
| Depth Pro 原始輸出（未校正） | 0.871 m | 1.827 m | +0.956 m |

- 設定：影像以 `cv2.ROTATE_180` 旋轉（不是鏡射）。CA-DFDS 與深度模型的遮罩跟著旋轉（代表分割模型重新找到車）；深度模型用原本的線性校正。
  STMAP 沒有分割能力，取樣框留在原位置（與學長 `stmap_dfds_rotation_eval.py` 相同）。
- UniDepthV2 原始輸出旋轉後 RMSE 反而下降（1.88 → 1.61 m），是因為原本整體高估約 1.8 m、旋轉後估測值整體變小，偏差剛好抵銷；
  誤差標準差由 0.42 增為 1.22 m，穩定度其實變差。套用校正後（公平比較）RMSE 由 0.357 增為 3.013 m。
- CA-DFDS 重新計算了 480 張旋轉影像的 CIECAM02 特徵（`rotation_cadfds.npz`），估測距離最大變化 0.0 m：3×3 高斯與拉普拉斯對 180° 旋轉對稱，遮罩內的統計量不變，所以結果不受構圖影響。
- 深度模型推論：`baselines/demo_deep.py`（`results/demo/rotation_deep.csv`）。逐張結果：`rotation_per_image.csv`；彙總：`rotation_summary.json`。
- `rotation_pairs/red_<距離>_orig.jpg`／`_rot180.jpg`：同尺寸成對圖，可做網頁拖曳對照。

**② 灰階 vs CIECAM02 清晰度**

- `gray_vs_ciecam/<車色>_<距離>_gray.png`／`_ciecam.png`：同一裁切、同尺寸（1600×1200）；紅色與黑色車 × 1.2／3.7／6.7 m。兩張各自用遮罩內 99 百分位做色階上限。
- 480 張 RMSE（手繪遮罩）：灰階 0.362 m、CA-DFDS 0.337 m、只用 J 0.313 m。
  **注意**：改善主要來自明度 J（只用 J 已達 0.313 m），色度 a_c、b_c 的貢獻很小（RESULTS 補充 E、M）；黑色車上 CA-DFDS 改善最明顯（灰階 0.541 → 0.468 m）。

- 影片 `video.mp4`：紅色車 12 個距離的旋轉對照 → RMSE 卡 → 灰階／CIECAM02 拖曳滑桿（35.97 s）。`video_short.mp4`：13.2 s。
- 產生：`../.venv/Scripts/python.exe baselines/demo_deep.py` → `python scripts/make_demo_assets.py --only C-F1`
