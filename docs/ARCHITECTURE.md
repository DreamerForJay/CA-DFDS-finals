# CA-DFDS 整體架構與方法說明（2026-11-16）

給教授、學長、同學和協作 AI：這份文件說明 CA-DFDS 是什麼、整個流程怎麼走，以及和其他方法的比較。

## 1. 一句話說明

CA-DFDS（Color-Aware Depth From Defocus with Segmentation）：用**一台相機、一張照片**估車距。

- 相機設定：大光圈 f/1.4、對焦在 15 m（後對焦）。
- 原理：1–7 m 內，車越近越模糊。
- 作法：在**車輛遮罩**內量測 **CIECAM02 感知色彩空間**的清晰度，再用事先擬合的**校正曲線**換算成距離。

## 2. 流程圖

```
           ┌─────────────── 離線（做一次）────────────────┐
校正集影像 │ 白／灰／黑車，1.0–7.0 m，每 0.5 m，各 10 張     │
 + 遮罩 ──▶│ 每張算清晰度 σ → 每個距離取平均 → 擬合二次曲線 │──▶ 校正曲線 d = c2·σ² + c1·σ + c0
           └──────────────────────────────────────────────┘     （results/calibration_methods.json）

           ┌──────────────────────── 線上（每張影像）───────────────────────────────┐
輸入影像 ─▶│ ① 車輛遮罩 ─▶ ② 前處理 ─▶ ③ 色彩轉換 ─▶ ④ 清晰度 ─▶ ⑤ 查校正曲線 │─▶ 距離 (m)
(f/1.4)    └────────────────────────────────────────────────────────────────────────┘
```

| 步驟 | 內容 | 程式 |
|---|---|---|
| ① 車輛遮罩 | 標出影像中「車」的像素。**目前實驗用手繪遮罩**，GUI 另可用 SAM 自動分割，見第 3 節 | `get_mask.py`（學長）、GUI 的 SAM 框選 |
| ② 前處理 | 3×3 高斯濾波去雜訊；只取遮罩外框範圍計算，結果和整張相同但較快 | `cadfds_core.crop_to_mask` |
| ③ 色彩轉換 | sRGB → XYZ → **CIECAM02**（D65、L_A = 2000/π×0.2、Y_b = 20、Average），得到明度 J，以及色度 a_c = C·cos h、b_c = C·sin h | `cadfds_core.ciecam_channels` |
| ④ 清晰度 | 三個通道各做 3×3 拉普拉斯，加權相加 w_J·∇²J + w_a·∇²a_c + w_b·∇²b_c，在遮罩內取標準差得到 σ（論文式 (5)） | `cadfds_core.sharpness` |
| ⑤ 換算距離 | σ 代入二次校正曲線得到距離（cm） | `np.polyval(coef, σ)` |

- 論文權重是 (w_J, w_a, w_b) = (0.8136, 0.1782, 0.0081)。
- 實作上先存三通道拉普拉斯的 3×3 共變異矩陣 Σ，σ(w) = √(wᵀΣw)，換權重時不必重算影像。

**為什麼用 CIECAM02 而不是灰階？**
灰階清晰度會受車身顏色影響：例如黑色、藍色車在灰階下對比低，同距離的清晰度和白車不同。
CIECAM02 把明度和色度分開，比較接近人眼感知，目標是讓**不同車色共用同一條校正曲線**。

## 3. 遮罩：SAM 還是 YOLO？

**結論：目前 CA-DFDS 的實驗數據（表 8、19 色、demo 影片）用的是「手繪遮罩」，不是 SAM，也不是 YOLO。**

| | 用途 | 輸出 | 本專案狀態 |
|---|---|---|---|
| 手繪遮罩 | 學長用 `get_mask.py` 以滑鼠塗出車輛範圍 | 像素遮罩 `.npy` | **目前實驗使用**。測試集各車色共用 `firetruck_black/..._mask/T1–T12.npy`，校正集用 `training/Training_dataset_mask` |
| SAM（Segment Anything） | 給一個點或一個框，自動切出物體的**像素輪廓** | 像素遮罩 | 前一篇 DFDS 論文用 SAM。本專案 GUI 新增「SAM 框選」（SAM ViT-B，GPU 約 0.1 秒）；以手繪遮罩外框當提示時，與手繪遮罩的 IoU 平均 **0.945**（最低 0.90，480 張） |
| YOLO（偵測版） | 自動找出「車在哪裡」 | **矩形框**，不是遮罩 | 尚未使用 |
| YOLO-seg（分割版） | 自動找車並給像素遮罩 | 像素遮罩 | 尚未使用 |

- **YOLO 不是「不需要遮罩」**。CA-DFDS 必須知道「哪些像素是車」才能算清晰度：
  - 若只用矩形框，框內會混到背景（路面、建築），背景的模糊程度和車不同，σ 會被污染。
  - 所以要嘛用 YOLO-seg 直接出遮罩，要嘛用「**YOLO 找框 → SAM 依框切出遮罩**」。
- **SAM 需要提示**（點或框）。給一個點常只切出車門一小塊（IoU 約 0.3），給框才能切出整個車尾（IoU 約 0.95）。
- 現場全自動的建議流程：**YOLO 偵測車輛框 → SAM 依框分割 → CA-DFDS 算距離**，見 [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) C3。
  - 目前 GUI 的「SAM 框選」等於用人手取代 YOLO 那一步。

## 4. 和其他方法比較（2026-11-16，同一批 480 張影像）

**設定**：

- 測試集：紅、綠、藍、黑四色消防車，距離 1.2–6.7 m（T01–T12），每個距離 10 張，共 480 張。
- 遮罩：各方法都使用同一份手繪遮罩。
- 指標：每張影像各自估測（per-image 協定）。

| 方法 | 類別 | 紅 | 綠 | 藍 | 黑 | **整體 RMSE** | MAE | 平均偏差 | 每張時間 |
|---|---|---|---|---|---|---|---|---|---|
| CA-DFDS（論文權重） | 本研究 | 0.276 | 0.196 | 0.348 | 0.468 | **0.337** | 0.270 | +0.056 | 2264 ms（CPU） |
| CA-DFDS 重新最佳化（式 5） | 本研究變體 | 0.243 | 0.184 | 0.304 | 0.456 | **0.314** | 0.246 | +0.044 | 2264 ms（CPU） |
| CA-DFDS J only | 本研究變體 | 0.227 | 0.191 | 0.307 | 0.458 | **0.313** | 0.246 | +0.043 | 2264 ms（CPU） |
| 灰階 DFD | 傳統基準 | 0.216 | 0.174 | 0.393 | 0.541 | **0.362** | 0.263 | +0.102 | 83 ms（CPU） |
| UniDepth V2（ViT-S） | 他人方法 | 1.713 | 2.096 | 1.705 | 1.974 | **1.879** | 1.832 | +1.832 | 26 ms（GPU） |
| ML-Depth-Pro | 他人方法 | 0.603 | 0.924 | 1.068 | 0.821 | **0.871** | 0.678 | −0.380 | 268 ms（GPU） |

單位：公尺。

- 來源：`results/method_comparison.{md,json}`，以及逐張結果 `results/method_comparison_per_image.csv`。
- 產生方式：
  - 深度模型：`baselines/eval_deep.py`
  - 校正曲線：`scripts/build_calibrations.py`
  - 合併：`scripts/compare_methods.py`

**解讀**：

1. CA-DFDS 的誤差約 0.3 m。UniDepth V2 約 1.9 m，Depth Pro 約 0.9 m。
   - 在這個 1–7 m 的近距離情境，CA-DFDS 比通用深度模型準 2.6–5.6 倍。
2. CA-DFDS 對比灰階 DFD：
   - 整體 RMSE 0.337 對 0.362。
   - 優勢主要在**低對比的藍、黑色車**：藍 0.348 對 0.393，黑 0.468 對 0.541。
   - 紅、綠車則是灰階略好。
3. 深度模型的誤差主要是**系統性偏差**：
   - UniDepth 每張平均高估 1.83 m。
   - Depth Pro 近距離（1.2 m）和遠距離（6.7 m）誤差大，中段（3.7 m）只有 0.23 m。
4. 公平性說明（給教授報告時請一併說明）：
   - 依論文作法，深度模型使用官方權重，**不 retrain、不校正**；CA-DFDS 則有用 CARLA 校正集擬合曲線。
   - 深度模型在 CARLA 合成影像上屬於跨領域使用。
   - 若給它們也做線性校正，誤差會下降。這是可以補做的對照實驗。
5. 時間：
   - CA-DFDS 是 CPU 單執行緒、未優化（主要花在 colour-science 的 CIECAM02 轉換）；改用 GPU 或查表可大幅加速（改善計畫 D 類）。
   - 深度模型的數字是 RTX 5080 上只計推論的時間。
6. 數據警示：校正用的黑色車影像已於 2026-07-31 重新產生，論文校正曲線無法以現有資料重現（[LAB_REPORT_2026-11-15.md](LAB_REPORT_2026-11-15.md) 第 4 節）。以上數字全部基於現有資料。

## 5. 硬體與環境

| 項目 | 實驗室主機 |
|---|---|
| CPU | Intel Core Ultra 7 265K（20 執行緒）— CA-DFDS、灰階 DFD |
| GPU | NVIDIA RTX 5080 16 GB（Blackwell、sm_120、CUDA 12.8）— SAM、UniDepth、Depth Pro |
| Python 環境 | `.venv-cadfds`：CA-DFDS 與 GUI，含 colour-science 0.4.6<br>`D:\Carla-0.10.0-Win64-Shipping\.venv`：torch 2.11 + cu128、segment_anything、UniDepth、Depth Pro |

- GUI 啟動時會自動偵測並顯示 CPU、GPU 型號與 VRAM。換到 RTX 3060 等其他電腦也適用。
- 舊報告（paper/datacopy/06）說 UniDepth、Depth Pro 在 5080 上很慢。改用 torch 2.11 + cu128 後已正常：UniDepth 約 30 ms、Depth Pro 約 270 ms。

## 6. 檔案對照

| 檔案 | 角色 |
|---|---|
| `cadfds_core.py` | 核心演算法：CIECAM02、清晰度、校正 |
| `extract_features.py` / `evaluate.py` / `reoptimize.py` | 批次特徵抽取、評估、權重最佳化 |
| `gui_demo.py` + `run_gui.bat` | 現場 demo、方法比較 GUI（[GUI_DEMO.md](GUI_DEMO.md)） |
| `baselines/deep_worker.py` | SAM／UniDepth／Depth Pro 的 GPU 背景程式（GUI 自動呼叫） |
| `baselines/eval_deep.py` | 深度模型批次評估 |
| `scripts/build_calibrations.py` / `scripts/compare_methods.py` | 各方法校正曲線、整體比較表 |
