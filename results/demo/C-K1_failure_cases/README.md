# C-K1 失敗案例

Each image has a one-line cause (`captions.json`, for the web gallery).

- `worst_beyond_7.2m.jpg`：超出校正範圍（7.2 m）：校正曲線只涵蓋 1–7 m，外插誤差變大。
- `worst_exposure_x0.5.jpg`：曝光 ×0.5（偏暗）：紋理對比下降，σ 變小，距離被高估或低估。
- `worst_exposure_x0.7.jpg`：曝光 ×0.7：同上，程度較輕。
- `worst_exposure_x1.4.jpg`：曝光 ×1.4（偏亮）：亮部飽和失去紋理。
- `worst_exposure_x2.0.jpg`：曝光 ×2.0：大面積過曝，清晰度量測失準。
- `worst_flipped.jpg`：上下鏡射影像（資料集 reverse）：與旋轉不同，說明見 RESULTS 補充 Q。
- `worst_suv_runA.jpg`：換車種（SUV）未重新校正：車身紋理不同，消防車的校正曲線不適用。
- `worst_suv_runB.jpg`：換車種（SUV，第二次渲染）：同上。
- `mask_error_box_iou0.33.png`：遮罩錯誤：框沒框好（與手繪遮罩 IoU 0.33），清晰度混入背景，誤差變大。

- 原始資料與數值：`results/failure_cases/`（`summary.csv`、`per_image.csv`）、RESULTS 補充 Q。逆光情境尚未量測。
