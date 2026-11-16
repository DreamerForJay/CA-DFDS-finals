# 實驗結果紀錄

（依 README 第 5 節更新。完整流程、發現與建議請見 [docs/LAB_REPORT_2026-11-15.md](docs/LAB_REPORT_2026-11-15.md)。）

## 共通資訊

- 執行日期／電腦：2026-11-15，實驗室 RTX 5080 主機（DESKTOP-GOLKTML；Intel Core Ultra 7 265K、64 GB RAM、NVIDIA GeForce RTX 5080）
- 環境：Python 3.12.7 venv（`.venv-cadfds`），`numpy 1.26.4`、`colour-science 0.4.6`、`opencv-python`、`scipy`；ffmpeg 使用 `imageio-ffmpeg` 附帶之執行檔
  - README 建議 conda + Python 3.10；本機無 conda，改用 venv，套件版本符合 `requirements.txt`
- 資料根目錄 `<DATA>` = `D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas`
- 校正集：`<DATA>/training/firetruck_<色>/Test_dataset_Focus1500cm_f1.4_f22.0`（1.0–7.0 m，13 距離 × 10 張），遮罩 `<DATA>/training/Training_dataset_mask`
- 測試集：`<DATA>/firetruck_<色>/Test_dataset_Focus1500cm_f1.4_f22.0`（資料夾內為 1.2–7.2 m 共 13 組；**遮罩只有 T1–T12，即 1.2–6.7 m**，T13 = 7.2 m 自動略過，與 README 所述 12 組一致）
- 測試遮罩：有自己 `_mask` 資料夾的顏色優先用自己的，其餘用 `<DATA>/firetruck_black/Test_dataset_Focus1500cm_f1.4_f22.0_mask`。已檢查所有 14 個現存測試遮罩資料夾 T1–T12 與 black 版本 **IoU 皆為 1.00（完全相同）**，共用不影響結果
- 表中 RMSE 皆為「逐張」協定（單位 m）；逐距離協定數值幾乎相同（差 ≤ 0.0011 m），完整見 CSV

> ⚠️ **重要：校正用黑色車影像已於 2026-07-31 重新產生**，與論文／`results/selftest.csv` 當時使用之影像不同。
> 因此本次以現有資料得到之 `opt` 校正曲線為 d = −6.530σ² + 125.95σ + 69.80（cm），R² = 0.942，
> 而非論文之 d = −6.562σ² + 128.83σ + 68.39，R² = 0.969。詳見報告第 4 節。

---

## 任務 A｜抽取特徵

- 指令：`bash logs/run_taskA.sh`（逐色呼叫 `extract_features.py --workers 16`）
- 資料路徑：見上方共通資訊
- 結果：28 組特徵全部成功

| 類別 | 顏色 | 每組有效影像 |
|---|---|---|
| 校正（`cal_*`） | white, gray, black, red, green, blue | 130（13 距離 × 10） |
| 測試（`test_*`） | 論文 19 色：red, green, blue, black, gray, deepGray, deep160Gray, deepRed, deepGreen, deepBlue, brown, chartreuse, cyan, magenta, orange, rose, springGreen, violet, yellow；另加 azure, pureBlue, firetruck（原色） | 120（12 距離 × 10） |

- 檔案：`features/*.npz`（已 `git add -f`，共 2.4 MB）、`logs/taskA.log`
- 備註：每個測試集 T13（7.2 m）10 張因無遮罩被略過（預期行為）；改善計畫 B4 提到之損毀影像 `black/T01/R/D1.0_shot03.png` 現已可讀（黑色校正集已重新產生）

## 任務 B｜表 8（測試距離 1.2–6.7 m）

- 指令：
  ```bash
  python evaluate.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz \
    --test features/test_red.npz features/test_green.npz features/test_blue.npz features/test_black.npz \
    --extra "reopt5=0.9532,0.0468,0" "Jonly=1,0,0" --out results/table8_testdist.csv
  python reoptimize.py --calib ... --form 5 --out results/reopt_eq5.json
  python reoptimize.py --calib ... --form 7 --out results/reopt_eq7.json
  ```
- 結果（校正：白＋灰＋黑；RMSE，m，逐張）：

| 清晰度設定 | 校正 R² | 紅 | 綠 | 藍 | 黑 | 平均 |
|---|---|---|---|---|---|---|
| 灰階基準 | 0.934 | **0.216** | **0.174** | 0.393 | 0.541 | 0.331 |
| w1 (0.6, 0.2, 0.2) | 0.915 | 0.311 | 0.235 | 0.801 | 0.609 | 0.489 |
| w2 (0.8, 0.1, 0.1) | 0.939 | 0.269 | 0.192 | 0.502 | 0.491 | 0.364 |
| opt (0.8136, 0.1782, 0.0081)（論文） | 0.942 | 0.276 | 0.196 | 0.348 | 0.468 | 0.322 |
| reopt5 (0.9532, 0.0468, 0) | 0.946 | 0.240 | 0.185 | **0.304** | **0.456** | **0.296** |
| J only (1, 0, 0) | 0.946 | 0.227 | 0.191 | 0.307 | 0.458 | **0.296** |

- 重新最佳化（`reoptimize.py`，起始 (0.6, 0.2, 0.2)，w ≥ 0、Σw = 1，SLSQP）：

| 形式 | 權重 (J, a_c, b_c) | 輪數 | 校正 RMSE |
|---|---|---|---|
| 式 (5) 直接最佳化 | (0.9434, 0.0566, 0.0000) | 48 | 43.48 cm |
| 式 (7) 替代特徵 | (1.0000, 0.0000, 0.0000) | 7 | 43.66 cm |

- 檔案：`results/table8_testdist.csv`、`results/reopt_eq5.json`、`results/reopt_eq7.json`、`results/calibration_opt.json`（已改為本次資料之曲線；舊版見 commit 285bd34）、`logs/taskB.log`
- 備註：
  1. 測試距離上，論文權重 opt 只在藍、黑兩色優於灰階，在紅、綠兩色反而較差；4 色平均 0.322 vs 0.331 m，差距很小
  2. 兩種重新最佳化都把權重集中到 J（a_c ≤ 0.057、b_c = 0），J only 與 reopt5 表現幾乎相同 → 支持「改善主要來自 CIECAM02 明度 J」
  3. `reopt5` 使用 README 給定之 (0.9532, 0.0468, 0)；本機以現有資料重算得 (0.9434, 0.0566, 0)，差異來自黑色校正影像重新產生

## 任務 C｜決賽 demo 影片

- 指令：`python make_demo_video.py --calibration results/calibration_opt.json --features features/test_red.npz features/test_green.npz features/test_blue.npz --labels 紅色車 綠色車 藍色車 --mask_dir <DATA>/firetruck_black/Test_dataset_Focus1500cm_f1.4_f22.0_mask --out results/CA-DFDS_demo_testdist.mp4`
- 結果：1920×1080、30 fps、H.264，4.7 MB；每色 12 距離 × 1 張，由遠到近

| 車色 | RMSE（12 張） |
|---|---|
| 紅色車 | 0.29 m |
| 綠色車 | 0.20 m |
| 藍色車 | 0.34 m |

- 檔案：`results/CA-DFDS_demo_testdist.mp4`、`logs/taskC.log`
- 備註：已擷取畫面確認中文字型（微軟正黑體）顯示正常；影片使用本次之校正曲線

## 任務 D｜處理時間

- 指令：`python timing.py --dataset <DATA>/firetruck_red/Test_dataset_Focus1500cm_f1.4_f22.0 --mask_dir <DATA>/firetruck_red/Test_dataset_Focus1500cm_f1.4_f22.0_mask --n 30 --out results/timing.json`
- 結果（30 張，含讀圖、讀遮罩、清晰度、校正曲線；不含遮罩取得；**僅 CPU 運算，未使用 GPU**）：

| 版本 | 平均 | 中位數 | FPS |
|---|---|---|---|
| 灰階 | 81.98 ms | 83.31 ms | 12.2 |
| CIECAM02（最終版本） | 2263.39 ms | 2264.38 ms | 0.44 |

- 檔案：`results/timing.json`、`logs/taskD.log`
- 備註：CIECAM02 約為灰階之 27.6 倍；量測時無其他計算工作。此主機即實驗室 RTX 5080 主機，但無法確認原 74.41 ms 是否為相同量測流程（本次 81.98 ms 包含 2048×1536 PNG 讀檔）

## 任務 E｜19 種車色 × 最佳化權重

- 指令：`python evaluate.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz --test features/test_*.npz --extra "reopt5=0.9532,0.0468,0" "Jonly=1,0,0" --out results/19colors_opt.csv`
- 結果（RMSE，m，逐張；粗體為該色最佳）：

| 車色 | 灰階 | w1 | w2 | opt（論文） | reopt5 | J only |
|---|---|---|---|---|---|---|
| red | **0.216** | 0.311 | 0.269 | 0.276 | 0.240 | 0.227 |
| green | **0.174** | 0.235 | 0.192 | 0.196 | 0.185 | 0.191 |
| blue | 0.393 | 0.801 | 0.502 | 0.348 | **0.304** | 0.307 |
| black ¹ | 0.541 | 0.609 | 0.491 | 0.468 | **0.456** | 0.458 |
| gray ¹ | 0.366 | 0.441 | 0.378 | 0.384 | 0.367 | 0.368 |
| deepGray | **0.212** | 0.239 | 0.219 | 0.224 | 0.214 | 0.213 |
| deep160Gray | 0.304 | 0.361 | 0.294 | 0.285 | **0.270** | 0.272 |
| deepRed | 0.386 | 0.437 | 0.372 | 0.393 | 0.347 | **0.340** |
| deepGreen | 0.244 | 0.315 | 0.233 | 0.247 | 0.206 | **0.202** |
| deepBlue | 0.362 | 0.611 | 0.395 | **0.267** | 0.272 | 0.285 |
| brown | 0.401 | 0.475 | 0.348 | 0.325 | **0.314** | 0.315 |
| chartreuse | 0.311 | 0.333 | 0.282 | **0.267** | 0.281 | 0.279 |
| cyan | 0.313 | 0.311 | 0.284 | **0.269** | 0.280 | 0.281 |
| magenta | **0.222** | 0.326 | 0.268 | 0.289 | 0.257 | 0.246 |
| orange | **0.354** | 0.503 | 0.447 | 0.459 | 0.407 | 0.392 |
| rose | **0.404** | 0.547 | 0.477 | 0.498 | 0.455 | 0.440 |
| springGreen | 0.500 | 0.570 | 0.484 | 0.476 | 0.467 | **0.466** |
| violet | **0.330** | 0.390 | 0.377 | 0.495 | 0.415 | 0.392 |
| yellow | 0.280 | 0.302 | 0.248 | 0.242 | 0.244 | **0.241** |
| **19 色平均** | 0.332 | 0.427 | 0.345 | 0.337 | 0.315 | **0.311** |
| 17 色平均（未參與校正） | 0.318 | 0.416 | 0.335 | 0.327 | 0.303 | **0.299** |
| *azure（額外）* | 0.264 | 0.651 | 0.365 | 0.249 | 0.206 | 0.211 |
| *pureBlue（額外）* | 0.294 | 0.702 | 0.404 | 0.270 | 0.224 | 0.230 |
| *firetruck 原色（額外）* | 0.436 | 0.502 | 0.431 | 0.424 | 0.415 | 0.416 |

¹ black、gray 之車色參與校正（距離不同），不屬於「未參與校正」之新車色。

- 19 色中優於灰階之車色數：opt 9／19、reopt5 11／19、J only 11／19
- 檔案：`results/19colors_opt.csv`、`logs/taskE.log`

## 任務 F｜學長原始程式

- 找到：`Carla-0.10.0-Win64-Shipping/test_ciecam.ipynb`（2026-07-31），已複製到 `results/original_code/`，並把各 cell 匯出成 `.py` 方便閱讀（原檔未修改）
- 搜尋範圍：D 槽全部、使用者桌面／文件／下載（排除套件與 CARLA 引擎資料夾）；關鍵字 SLSQP、scipy.optimize、CIECAM02、tkinter/PyQt/gradio/streamlit

1. **權重交替最佳化**（cell 4，`test_ciecam_cell4_weight_alternating_optimization.py`）
   - 呼叫 `scipy.optimize.minimize(objective, w, bounds=[(0.01, 1)]*3)`，**未指定 method → SciPy 預設為 L-BFGS-B，不是 SLSQP**
   - 權重限制：**每個權重下限 0.01（非 0）、上限 1**；**總和為 1 不是用限制式**，而是在目標函數內 `w/Σw` 正規化，每輪結束再正規化一次
   - 停止條件：`np.allclose(w, new_w, atol=1e-5)` 或 **最多 10 輪**
   - 目標函數：**逐距離平均**（每距離 10 張先平均各通道之拉普拉斯標準差）後之校正 RMSE（cm）
   - 特徵形式：**式 (7) 替代特徵**（各通道標準差之線性組合 X·w），不是式 (5)
   - 起始 (0.6, 0.2, 0.2)，校正色 white、gray、black
   - notebook 留存之輸出：第 2 輪 (0.816, 0.176, 0.008) ≈ 論文權重 (0.8136, 0.1782, 0.0081)；最終收斂於 (0.9804, 0.0098, 0.0098)，a_c、b_c 卡在下限 0.01
2. **測試／推論程式**（cell 2、3）：**確認使用式 (5)**，即先加權 `w_j·lap_J + w_ac·lap_ac + w_bc·lap_bc` 再取遮罩內標準差；每距離 10 張先平均清晰度再代入曲線（逐距離協定）
3. **近距離版本 GUI（作品說明書圖 5）**：**未找到**。唯一候選為 `old/test_carla.ipynb`（已複製為 `results/original_code/test_carla_candidate_gui.ipynb`），是 OpenCV 視窗、滑鼠點 4 點框選之雙光圈灰階 DFD 工具（Focus 50 cm），不是 tkinter／Qt 介面，無法確認是否為圖 5

## 任務 G｜現場 GUI 測試（學長舊版 GUI）

> 2026-11-16 起，此版改名為 `gui_old_senior.py`（學長舊版），截圖改名為 `gui_old_senior_*.png`；新版為 `gui_new_v3.py`。

- 指令：`python gui_old_senior.py --calibration results/calibration_opt.json`；截圖由 `logs/gui_screenshots.py` 自動操作同一個 GUI（代替檔案對話框選檔 → 載入遮罩 → 按「估測距離」→ 擷取視窗）
- 結果：

| 截圖 | 影像 | 清晰度 σ | 估測 | 真實 | 誤差 | 計算時間 |
|---|---|---|---|---|---|---|
| `results/gui_old_senior_1.png` | 紅色車 T05 shot01 | 2.09 | 3.04 m | 3.20 m | −0.16 m | 3020 ms |
| `results/gui_old_senior_2.png` | 藍色車 T09 shot01 | 5.55 | 5.68 m | 5.20 m | +0.48 m | 2115 ms |
| `results/gui_old_senior_3.png` | 綠色車 T02 shot01 | 0.58 | 1.41 m | 1.70 m | −0.29 m | 2164 ms |

- 檔案：`results/gui_old_senior_*.png`、`logs/gui_screenshots.py`、`logs/taskG.log`
- 備註：GUI 開啟、載入遮罩、估測、顯示真實距離與誤差皆正常；滑鼠塗抹遮罩功能未在本次截圖中測試

## 補充 H｜方法比較：CA-DFDS 對比灰階 DFD、UniDepth V2、ML-Depth-Pro（2026-11-16）

- 設定：紅、綠、藍、黑四色 × T01–T12（1.2–6.7 m）× 10 張，共 480 張。所有方法使用同一份手繪遮罩，每張影像各自估測。
- 深度模型：官方權重，不重新訓練、不校正；取遮罩內深度的中位數，並給定相機內參 fx = 1137.8 px。

| 方法 | 紅 | 綠 | 藍 | 黑 | 整體 RMSE (m) | 平均偏差 (m) | 每張時間 |
|---|---|---|---|---|---|---|---|
| CA-DFDS（論文權重） | 0.276 | 0.196 | 0.348 | 0.468 | **0.337** | +0.056 | 2264 ms CPU |
| CA-DFDS 重新最佳化（式 5） | 0.243 | 0.184 | 0.304 | 0.456 | **0.314** | +0.044 | 2264 ms CPU |
| CA-DFDS J only | 0.227 | 0.191 | 0.307 | 0.458 | **0.313** | +0.043 | 2264 ms CPU |
| 灰階 DFD | 0.216 | 0.174 | 0.393 | 0.541 | **0.362** | +0.102 | 83 ms CPU |
| UniDepth V2 | 1.713 | 2.096 | 1.705 | 1.974 | **1.879** | +1.832 | 26 ms GPU |
| ML-Depth-Pro | 0.603 | 0.924 | 1.068 | 0.821 | **0.871** | −0.380 | 268 ms GPU |

- SAM ViT-B 框選：以手繪遮罩外框當提示，與手繪遮罩的 IoU 平均 0.945、最低 0.902；每張約 110 ms（GPU）。
- 檔案：
  - 結果：`results/method_comparison.{md,json}`、`results/method_comparison_per_image.csv`、`results/deep_baselines.csv`、`results/calibration_methods.json`
  - 紀錄：`logs/deep_baselines.log`
- 解讀與公平性說明見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 4 節。

## 補充 I｜GUI v2 → v3（2026-11-16）

- v2 截圖已由 v3 取代，v2 的檔案已刪除。
- v3 說明見 [docs/GUI_DEMO.md](docs/GUI_DEMO.md)，截圖為 `results/gui_new_v3_*.png`；程式為 `gui_new_v3.py`。

## 補充 J｜深度模型線性校正，以及全自動遮罩（YOLO11 → SAM）（2026-11-16）

- **深度模型線性校正**：用和 CA-DFDS 相同的校正集（白、灰、黑 × 1.0–7.0 m × 10 張，共 390 張）擬合 d = a·預測 + b。
  - UniDepth：0.910·預測 − 1.43 m，R² 0.896。
  - Depth Pro：1.183·預測 − 0.16 m，R² 0.909。
- **全自動遮罩**：
  - 流程：YOLO11m 偵測 car/bus/truck，取最大框 → SAM ViT-B 依框分割。
  - YOLO 沒偵測到時，用近距離後備：整幅框和中下方框各跑一次 SAM，取分數高者。
  - YOLO 偵測率：1.2–2.7 m 幾乎為 0，3.2 m 為 50%，3.7 m 以上 100%。480 張中 170 張使用後備。
  - 和手繪遮罩的 IoU：平均 0.906（YOLO 0.925、後備 0.873），最低 0.660。
  - 時間：YOLO 約 13 ms、SAM 約 105 ms（GPU）。

| 方法 | 手繪遮罩 RMSE (m) | 全自動遮罩 RMSE (m) |
|---|---|---|
| CA-DFDS（本研究） | 0.337 | **0.310** |
| CA-DFDS 重新最佳化 | 0.314 | 0.289 |
| CA-DFDS 只用 J | 0.313 | 0.290 |
| 灰階 DFD | 0.362 | 0.325 |
| UniDepth V2（原始／線性校正） | 1.879／**0.357** | 1.883／0.355 |
| ML-Depth-Pro（原始／線性校正） | 0.871／0.751 | 0.870／0.750 |

**各距離 RMSE（手繪遮罩）**：

- CA-DFDS 在 1.2–3.7 m 為 0.14–0.33 m，UniDepth（線性校正）為 0.35–0.55 m。
- 4.2–6.7 m 則是 UniDepth 較好：0.12–0.22 m，CA-DFDS 為 0.26–0.51 m。
- 兩者互補，完整表格見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 4.3 節。

**結論**：

- 公平比較（深度模型也校正）時，UniDepth 和 CA-DFDS 的整體準確度相當。CA-DFDS 的優勢在近距離、可解釋，以及不需要 GPU。
- 全自動遮罩不會讓 CA-DFDS 變差，現場可以不用人工遮罩。

**檔案**：

- 結果：`results/deep_baselines.csv`（測試集，含自動遮罩欄位）、`results/deep_calib.csv`（校正集）、`results/calibration_methods.json`、`results/method_comparison.*`
- 特徵：`features/test_auto_*.npz`
- 紀錄：`logs/deep_baselines.log`、`logs/deep_calib.log`
- 自動遮罩 `.npy` 在 `logs/scratch/auto_masks/`，未提交，可用 `baselines/eval_deep.py --auto` 重新產生。

---

# 2026-11-16 GitHub issue 實驗（#1–#8）

共通：實驗室 RTX 5080 主機（DESKTOP-GOLKTML；Intel Core Ultra 7 265K、64 GB RAM、NVIDIA GeForce RTX 5080），`.venv-cadfds`（Python 3.12.7、numpy 1.26.4、colour-science 0.4.6）與 `../.venv`（torch 2.11.0+cu128、CUDA 12.8）。
除非另外說明，校正集為白＋灰＋黑（1.0–7.0 m），RMSE 為逐張協定（單位 m）。黑色校正影像仍為 2026-07-31 重新產生的版本（見共通資訊的警告）。

## 補充 K｜式 (5) 與式 (7) 清晰度定義對照（issue #3／M1）

- 指令：`python scripts/m1_eq5_vs_eq7.py --out results/eq5_vs_eq7.csv`
- 方法：用 `features/*.npz` 的共變異矩陣 Σ。式 (5) σ = √(wᵀΣw)，式 (7) S = w·√diag(Σ)。同一組權重、同一校正集，各自擬合二次曲線，再測 19 色。

| 權重 (J, a_c, b_c) | 式 | 校正 R² | 紅 | 綠 | 藍 | 黑 | 19 色平均 | 17 色平均（未參與校正） |
|---|---|---|---|---|---|---|---|---|
| 論文 opt (0.8136, 0.1782, 0.0081) | (5) | 0.942 | 0.276 | 0.196 | 0.348 | 0.468 | 0.337 | 0.327 |
| 論文 opt (0.8136, 0.1782, 0.0081) | (7) | 0.923 | 0.194 | 0.202 | 0.446 | 0.575 | 0.316 | 0.294 |
| w1 (0.6, 0.2, 0.2) | (5) | 0.915 | 0.311 | 0.235 | 0.801 | 0.609 | 0.427 | 0.416 |
| w1 (0.6, 0.2, 0.2) | (7) | 0.837 | 0.249 | 0.269 | 0.737 | 0.938 | 0.461 | 0.423 |
| w2 (0.8, 0.1, 0.1) | (5) | 0.939 | 0.269 | 0.192 | 0.502 | 0.491 | 0.345 | 0.335 |
| w2 (0.8, 0.1, 0.1) | (7) | 0.918 | 0.210 | 0.209 | 0.533 | 0.605 | 0.344 | 0.323 |
| reopt5 (0.9532, 0.0468, 0) | (5) | 0.946 | 0.240 | 0.185 | 0.304 | 0.456 | 0.315 | 0.303 |
| reopt5 (0.9532, 0.0468, 0) | (7) | 0.942 | 0.199 | 0.189 | 0.338 | 0.476 | 0.305 | 0.291 |
| J only (1, 0, 0) | (5) = (7) | 0.946 | 0.227 | 0.191 | 0.307 | 0.458 | 0.311 | 0.299 |

- 結論：
  1. 兩式只在 a_c、b_c 權重不為 0 時不同；J only 時完全相同。
  2. 論文權重下，式 (7) 的 19 色平均較好（0.316 vs 0.337 m），但校正 R² 較低（0.923 vs 0.942），而且藍、黑兩色明顯變差（0.446／0.575 m）。兩式互有勝負，沒有一式全面較好。
  3. **建議作品說明書統一寫式 (5)**：推論程式（`test_ciecam.ipynb` cell 2、3）用的是式 (5)。並註明論文權重是用式 (7) 替代特徵最佳化得到的（任務 F）。
- 檔案：`results/eq5_vs_eq7.csv`、`scripts/m1_eq5_vs_eq7.py`

## 補充 L｜消融 E：校正集加入紅綠藍（issue #4／M2）

- 指令：`python scripts/m2_ablation_E.py --out results/ablation_E.csv`
- 方法：
  - 校正集改為白＋灰＋黑＋紅＋綠＋藍（`cal_*.npz`，1.0–7.0 m，780 張）。
  - 權重**只用校正集**重新最佳化，流程同 `reoptimize.py`（SLSQP、w ≥ 0、Σw = 1、起點 (0.6, 0.2, 0.2)）。
  - 測試集是 19 色扣除色相已在校正集內的 5 色（紅、綠、藍、黑、灰），**剩 14 色**（issue 寫 13 色，實際為 14 色）。
  - A–D 也在同一個 6 色校正集上重算，切分一致。
- 結果：
  - E：式 (5) 最佳化得 (0.9786, 0.0214, 0)，61 輪；式 (7) 得 (1, 0, 0)，9 輪。
  - 各輪權重：`results/ablation_E_weights.json`

| 模型 | 色彩特徵／權重 | 校正集 | 校正 R² | 14 色平均（未參與校正） | 5 色平均（紅綠藍黑灰） |
|---|---|---|---|---|---|
| A | 灰階 | 白灰黑 | 0.934 | 0.330 | 0.338 |
| B | CIECAM02 固定 (0.6, 0.2, 0.2) | 白灰黑 | 0.915 | 0.409 | 0.479 |
| C | CIECAM02 論文 opt | 白灰黑 | 0.942 | 0.338 | 0.334 |
| D | CIECAM02 只用 J | 白灰黑 | 0.946 | **0.312** | 0.310 |
| A | 灰階 | 6 色 | 0.954 | 0.320 | 0.344 |
| B | CIECAM02 固定 (0.6, 0.2, 0.2) | 6 色 | 0.917 | 0.404 | 0.476 |
| C | CIECAM02 論文 opt | 6 色 | 0.958 | 0.327 | 0.337 |
| D | CIECAM02 只用 J | 6 色 | 0.963 | **0.301** | 0.309 |
| **E** | CIECAM02 6 色重新最佳化 (0.9786, 0.0214, 0) | 6 色 | 0.963 | 0.303 | 0.310 |

- 結論：
  1. 校正集加入紅綠藍後，所有模型的 14 色平均都改善約 0.01 m，校正 R² 提高到 0.95–0.96。
  2. 即使校正集有彩色車，最佳化仍把權重幾乎全放在 J（a_c 0.02、b_c 0）。E 與 J only 幾乎相同（0.303 vs 0.301 m）。
  3. 「CIECAM02 的改善主要來自明度 J」的結論不受校正集組成影響。
- 檔案：`results/ablation_E.csv`、`results/ablation_E_weights.json`、`scripts/m2_ablation_E.py`

## 補充 M｜完整管線分段計時，含 GPU 版 CIECAM02（issue #1／M3）

- 指令：見 `scripts/m3_timing_full_pipeline.py` 檔頭。CPU 部分用 `.venv-cadfds`，GPU 與遮罩部分用 `../.venv`，最後 `--merge`。
- 新增 `cadfds_gpu.py`：用 PyTorch 實作 sRGB → CIECAM02 正向轉換、拉普拉斯與遮罩內共變異，公式逐項對照 colour-science 0.4.6（含其 signed power 與除以 0 的處理）。
  - 驗證：32 張（紅、藍、黑、黃）與 CPU 特徵比較，σ 相對誤差 fp64 為 1.8×10⁻¹³、fp32 為 2.6×10⁻⁶。
  - 30 張的距離預測與 CPU 版完全相同（RMSE 0.2764 m）。
- 設定：紅色車測試集 30 張，平均分布於 1.2–6.7 m；另有暖機 3 張，不計入。計時含讀 PNG（2048×1536）與讀遮罩。

| 版本 | 讀圖 | 遮罩 | 裁切 | 高斯 | 上傳 GPU | sRGB→CIECAM02 | 拉普拉斯＋共變異 | **總計（平均）** | FPS |
|---|---|---|---|---|---|---|---|---|---|
| CPU 整張（現況） | 49.6 | 4.6 | — | 1.6 | — | 2283.5 | 68.9 | **2408.1 ms** | 0.42 |
| CPU 遮罩外框裁切 | 49.6 | 4.5 | 15.3 | 1.2 | — | 1284.7 | 61.9 | **1417.3 ms** | 0.71 |
| GPU fp64 整張 | 52.1 | 5.5 | — | 2.7 | 8.9 | 17.8 | 5.7 | **92.9 ms** | 10.8 |
| GPU fp64 裁切 | 45.6 | 4.4 | 14.2 | 1.5 | 5.2 | 9.4 | 5.3 | **85.7 ms** | 11.7 |
| GPU fp32 整張 | 46.1 | 4.3 | — | 2.8 | 8.8 | 3.6 | 5.1 | **70.9 ms** | 14.1 |
| GPU fp32 裁切 | 45.8 | 4.4 | 14.2 | 1.6 | 5.1 | 2.3 | 5.0 | **78.3 ms** | 12.8 |

（單位 ms／張；曲線換算 < 0.1 ms，未列出。）

**自動遮罩（YOLO11m → SAM ViT-B，GPU；同 30 張）**

- YOLO 偵測：平均 12.9 ms。
- SAM（影像編碼＋框解碼）：平均 106.9 ms。
- 含讀檔、寫檔的總時間：平均 198.9 ms。
- 30 張中有 8 張用近距離後備框；與手繪遮罩的 IoU 平均 0.924。

結論：

1. CIECAM02 轉換改到 GPU 後，由 2283 ms 降為 17.8 ms（fp64），約快 128 倍；fp32 為 3.6 ms。
2. **不含遮罩時，GPU 版整條管線達 10.8–14.1 FPS，達到 M3「≥ 10 FPS」的目標**。現在的瓶頸是讀 PNG（約 50 ms）。
3. 含全自動遮罩時約 71 + 120 ≈ 190 ms／張，約 5 FPS；SAM 佔大部分時間。
4. GPU 版的裁切反而略慢：裁切本身要掃描整張遮罩（約 14 ms），省下的 GPU 時間比這還少。CPU 版裁切則省下約 1 秒。
5. 可行性頁可改寫為：「CIECAM02 版本在 RTX 5080 上約 71–93 ms／張（含讀圖，不含遮罩）」。

- 檔案：`results/timing_full_pipeline.json`、`cadfds_gpu.py`、`scripts/m3_timing_full_pipeline.py`

## 補充 N｜全自動遮罩：校正集也改用自動遮罩（issue #5／M4）

- 指令：
  ```bash
  ../.venv/Scripts/python.exe scripts/m4_auto_mask_calib.py        # 校正集 390 張自動遮罩
  .venv-cadfds/Scripts/python.exe scripts/m4_auto_mask_report.py    # 特徵、比較表、範例圖
  ```
- 遮罩方法同補充 J（YOLO11m → SAM ViT-B，YOLO 沒偵測到時用近距離後備框）。
- 三種協定：
  - hand：校正、測試都用手繪遮罩（論文）。
  - auto：校正用手繪、測試用自動遮罩（補充 J）。
  - fullauto：校正、測試都用自動遮罩，**完全不需要手繪遮罩**。

| 方法 | hand | auto | **fullauto** | 校正 R²（手繪／自動） |
|---|---|---|---|---|
| CA-DFDS（論文權重） | 0.337 | 0.310 | **0.317** | 0.942／0.950 |
| CA-DFDS 只用 J | 0.313 | 0.290 | **0.291** | 0.946／0.953 |
| 灰階 DFD | 0.362 | 0.325 | **0.331** | 0.934／0.941 |

（紅綠藍黑 × 1.2–6.7 m，各 480 張）

**遮罩品質（與手繪遮罩的 IoU）**

| 資料 | 張數 | 平均 IoU | 最低 IoU | YOLO 框 | 後備框（張數） |
|---|---|---|---|---|---|
| 測試 | 480 | 0.906 | 0.660 | 0.925 | 0.873（170） |
| 校正 | 390 | 0.892 | 0.750 | 0.916 | 0.852（150） |

- 時間：YOLO 約 13 ms，SAM 約 100–108 ms（GPU）。
- **各距離 RMSE（CA-DFDS，hand → fullauto）**：
  - 2.2 m 時由 0.207 變差為 0.327 m。原因是後備框把部分路面框進遮罩，見 `results/auto_mask_examples/black_T05_shot10_fallback.jpg`。
  - 4.2–5.2 m 時反而改善，由 0.48–0.51 降為 0.41–0.44 m。

結論：

1. 全自動流程（含校正）下，CA-DFDS 的 RMSE 為 0.317 m，比手繪遮罩的 0.337 m 還好。
2. 現場與校正都可以不用人工遮罩。
3. 主要弱點在 1.7–3.2 m：YOLO 認不出只露出車尾的車，必須靠後備框。

- 檔案：
  - 結果：`results/auto_mask.csv`（逐張：遮罩來源、IoU、時間、三種協定的預測）、`results/auto_mask_summary.json`、`results/auto_mask_examples/*.jpg`（綠＝手繪，紅＝自動）
  - 特徵：`features/cal_auto_*.npz`
  - 程式：`scripts/m4_auto_mask_*.py`
  - 遮罩 `.npy` 在 `logs/scratch/auto_masks/calib/`，未提交。

## 補充 O｜Gamma 一致性與 gamma 1.0 重測 19 色（issue #6／M5）

- 指令：`../.venv/Scripts/python.exe scripts/m5_gamma.py`

**1. 檢查現有資料是否混用 gamma**

- 方法：
  - 同距離下比較各色影像的背景像素（擴張後的遮罩之外，取下半部），與參考色（測試集用紅、校正集用白）比較。
  - 自動曝光只會讓亮度乘上一個增益，log–log 斜率約為 1。
  - gamma 1.0 與 2.2 混用時，斜率會約為 2.2 或 0.45。
- 結果：
  - 4 組距離（測試 T06、T12，校正 T07、T13）共 50 組比較，**斜率全部在 0.844–1.057**，相關係數 r 為 0.958–0.998。
  - 增益為 0.68–1.40，來自相機的 `exposure_mode = histogram` 自動曝光。
- **結論：現有影像沒有 gamma 1.0／2.2 混用**；車色之間的亮度差異來自自動曝光。
  - 作品說明書「部分深色／灰色影像為 gamma 1.0」的敘述，與現有檔案不符。
  - 可能原因：這些影像已用預設 gamma 重拍過（測試集 2026-01 至 04 月、黑色校正集 2026-07-31）。
  - 建議向學長確認後，刪除或修正該敘述。

**2. 統一改為 gamma 1.0（線性光）重測**

- 因為現有資料本來就一致，沒有需要「重產」的影像。改為把**所有**影像（校正 3 色＋測試 19 色）轉成 gamma 1.0 的 8-bit 影像，用來測試敏感度。
  - 轉換方式：I' = round(255 · EOTF_sRGB(I/255))
  - 特徵用 GPU 管線重算。
- 注意：這是由 8-bit sRGB 影像轉換而來，不是在 CARLA 中以 gamma 1.0 重新渲染。

| 設定 | 現有資料 | 全部 gamma 1.0 | 校正 2.2／測試 1.0（混用） |
|---|---|---|---|
| 灰階 DFD | 0.332（R² 0.934） | **0.251**（R² 0.970） | 0.330 |
| CA-DFDS 論文權重 | 0.337（R² 0.942） | **0.276**（R² 0.962） | 0.282 |
| CA-DFDS 只用 J | 0.311（R² 0.946） | **0.267**（R² 0.966） | 0.296 |

（19 色平均 RMSE，m；各色數值見 CSV）

觀察：

1. 統一改用線性光後，三種方法都明顯改善，校正 R² 提高到 0.96–0.97。灰階改善最多（−24%），在此設定下成為最佳。
   - 推測原因：散焦模糊在線性光強度上才是線性的，sRGB 編碼會扭曲模糊的量測。
   - 這可能影響「CIECAM02 優於灰階」的論點，**建議以 CARLA gamma 1.0 實際重新渲染確認**後再寫入論文。
2. 在 gamma 1.0 下，CIECAM02 管線會把已經是線性的影像再做一次 sRGB 解碼，因此對 CIECAM02 而言不是公平的設定。
   - 若要正式比較，CIECAM02 輸入端應改為不做解碼。目前尚未實作。
3. 混用（校正 2.2、測試 1.0）的 RMSE 沒有變差，反而接近或優於現有資料。這是誤差互相抵銷的偶然結果，不代表混用無害。

- 檔案：`results/gamma_check.csv`、`results/19colors_gamma_unified.csv`、`scripts/m5_gamma.py`；特徵 `features/g1_*.npz`

## 補充 P｜前期研究近距離比較表之量測硬體（issue #2）

- 資料來源：
  - 表中數字（STMAP 86.434、ML-Depth-Pro 1.273、UniDepth 0.353、DFDS 1.187 秒／張）出自 DFDS 論文 Table 3。
  - 論文：`D:\Carla-0.10.0-Win64-Shipping\paper\116.pdf`，第 4.4 節。
  - 簡報：`paper\DFDS.pdf`，第 18 頁 “Experimental Results -- Runtime”。
  - DFDS 論文作者：Ing-Chau Chang、Yu-Cheng Chen、Yu-Sen Wang、Chunghui Kuo、Chin-En Yen。
- 搜尋範圍：
  - 上述論文與簡報全文、`paper/宇森_中文初稿.docx`、`paper/datacopy/*.md`
  - 本機 D 槽中含 86.434／STMAP 的檔案
- 結果：**都沒有記載 CPU／GPU 型號**。
  - 論文第 4.1 節只寫相機為 Canon EOS-6D（4.0 cm、f/13 與 f/22）、資料為 20–110 cm、測試 200 張。
- `paper/datacopy/06_DFDS_CARLA重現結果報告.md` 第 5 點也指出：論文 STMAP 為 CPU 逐像素 48×48 卷積，重現版本的秒數不能直接比較。
- **結論**：硬體型號只能向論文作者（Yu-Cheng Chen 或指導教授）詢問。在取得之前，簡報上該表應註明「量測硬體未載明於原論文」，**不要寫成 RTX 5080**。

## 補充 Q｜其他車種與失敗案例（issue #8／M7、M8）

- 指令：`../.venv/Scripts/python.exe scripts/m78_vehicle_failure.py`（紀錄 `logs/m78.log`）
- 只用主機上**已有的**資料，沒有重新以 CARLA 渲染。
- 校正：消防車白＋灰＋黑。有手繪遮罩的資料用手繪遮罩的曲線；沒有手繪遮罩的資料（SUV、7.2 m）用自動遮罩的曲線（補充 N）。
- 特徵用 GPU 管線計算（與 CPU 相同）。

**1. 其他車種：SUV（nissan.patrol_2021）**

- ⚠️ `datas/audi.etron` 與 `datas/ford.mustang` 兩個資料夾**其實都是同一台白色 SUV**（2026-01-10 產生），只是兩次不同的渲染（逐像素不相同）。
  - 原因：`get_dataset2.py` 的車種清單中，CARLA 0.10 找不到 audi.etron、mustang，因此退回 nissan.patrol_2021。
  - 因此目前只有 1 個額外車種、1 種顏色，**未達 issue 要求的「轎車、SUV、貨車各 ≥ 3 色」**。補齊需要用 CARLA 重新產生資料。

| 資料 | 張數 | 距離 | 灰階 | CA-DFDS | J only | CA-DFDS 平均偏差 |
|---|---|---|---|---|---|---|
| 參考：消防車紅綠藍黑（自動遮罩） | 480 | 1.2–6.7 m | 0.331 | 0.317 | 0.291 | — |
| SUV 第 1 次渲染（audi.etron 資料夾） | 120 | 1.2–6.7 m | 1.237 | 1.269 | 1.112 | +1.10 m |
| SUV 第 2 次渲染（ford.mustang 資料夾） | 120 | 1.2–6.7 m | 1.249 | 1.275 | 1.115 | +1.10 m |

（RMSE，m）

- **各距離 CA-DFDS RMSE（第 1 次渲染）**：

  | 距離 | 1.2 | 1.7 | 2.2 | 2.7 | 3.2 | 3.7 | 4.2 | 4.7 | 5.2 | 5.7 | 6.2 | 6.7 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | RMSE (m) | 0.30 | 1.82 | 2.02 | 1.78 | 1.61 | 1.45 | 1.24 | 1.07 | 0.90 | 0.57 | 0.38 | 0.04 |

  幾乎全是**高估**。偏差隨距離遞減，不是固定位移，因此不是距離參考點不同造成的。
- 遮罩正常（YOLO 偵測、IoU 視覺確認），見 `results/failure_cases/worst_suv_runA.jpg`：真實 2.2 m，CA-DFDS 估 4.29 m、灰階估 4.20 m。
- **結論**：以消防車擬合的校正曲線**無法直接套用到其他車種**。
  - SUV 車尾比較平滑，同距離下的清晰度 σ 偏低，於是被判為較遠。灰階 DFD 也一樣。
  - 這是單張 DFD 的根本限制：σ 同時取決於散焦量與物體本身的紋理。
  - 論文應明確寫「校正曲線依車種／紋理而定」。
  - 改善方向：依車種分別校正（YOLO 類別 → 對應曲線），或改用雙光圈比值（f/1.4 與 f/22，資料夾 L/R 已有）來抵消紋理。

**2. 失敗案例**

| 案例 | 張數 | 灰階 | CA-DFDS | J only | CA-DFDS 偏差 | 說明 |
|---|---|---|---|---|---|---|
| 參考：消防車紅綠藍黑（手繪遮罩） | 480 | 0.362 | 0.337 | 0.313 | — | |
| 超出範圍 7.2 m（校正只到 7.0 m） | 40 | 0.531 | 0.480 | 0.456 | −0.47 m | 二次曲線在遠端趨平，會低估 |
| 上下翻轉影像（`datas/reverse`） | 480 | 0.362 | 0.337 | 0.313 | — | 與原圖**完全相同**：拉普拉斯與標準差對翻轉不變 |
| 曝光 ×0.5（線性光） | 240 | 0.595 | 0.790 | 0.750 | −0.62 m | 欠曝：σ 變小，判為較近 |
| 曝光 ×0.7 | 240 | 0.410 | 0.488 | 0.444 | −0.29 m | |
| 曝光 ×1.4 | 240 | 0.605 | 0.620 | 0.622 | +0.40 m | 遮罩內 21% 像素飽和 |
| 曝光 ×2.0 | 240 | 0.843 | 0.916 | 0.929 | +0.64 m | 遮罩內 47% 像素飽和 |

（曝光案例為紅、黑兩色；RMSE，m）

- **曝光**：只要曝光差 ±30%，RMSE 就從約 0.34 m 增加到 0.41–0.62 m。
  - CA-DFDS 在欠曝時比灰階更敏感（0.790 vs 0.595 m）。
  - 現場應固定曝光，或先依遮罩內亮度正規化 σ。
- **紋理**（19 色，f/22 影像遮罩內灰階拉普拉斯標準差）：
  - 紋理越多，誤差越小。Spearman ρ = −0.44，p = 0.058，19 色；僅接近顯著。
  - SUV 案例是更直接的低紋理證據。
- **強反光**：現有資料沒有對應場景，無法量測，填「—」。
- 檔案：
  - `results/multi_vehicle.csv`（SUV 各距離）
  - `results/failure_cases/summary.csv`、`per_image.csv`、`texture_by_color.csv`、`texture_stats.json`
  - `results/failure_cases/worst_*.jpg`（每個案例誤差最大的一張）

## 補充 R｜CARLA 重新產生多車種資料（M7）（2026-11-16）

- **資料**（`scripts/gen_multivehicle.py`、`scripts/run_multivehicle.sh`）：
  - 拍攝設定與 `get_dataset2.py` 相同：Town10HD、2048×1536、40 mm、對焦 1500 cm、f/1.4（R）／f/22（L）、每距離 10 張。
  - 明確指定車種，找不到即中止（不再 fallback）。
  - 新增 instance segmentation 相機，自動產生**精確車輛遮罩**（`*_mask/T<n>.npy`）。
  - 可改色的車種：`dodge.charger`（轎車）、`nissan.patrol`（SUV）、`ambulance.ford`（貨車，車尾有條紋與字樣，只有部分車身會變色）。`lincoln.mkz`、`carlacola`、`sprinter` 忽略 color 屬性，所以 carlacola 只拍原廠塗裝。
  - 測試色：紅、藍、黑，1.2–7.2 m。校正色：白、灰、黑，1.0–7.0 m。每 0.5 m 一個距離，共 13 個。
  - 合計 19 組 × 13 距離 × 10 張 × 2 光圈，存放於 `<DATA>/multivehicle/<車種>_<色>/`。
  - 已逐組檢查遮罩內平均色，所有顏色皆正確套用。
  - 貨車近距離時車尾會超出畫面上緣。
- **評估**（`scripts/m7_multivehicle_eval.py`，GPU 管線，二次校正曲線）：
  - 下表為整體 RMSE（m），三欄依序為 CA-DFDS 論文權重／灰階／只用 J。
  - 括號內為 CA-DFDS 的偏差（m）。

| 校正方式 | 消防車 | 轎車 charger | SUV patrol | 貨車 ambulance | 貨車 carlacola |
|---|---|---|---|---|---|
| A 消防車校正（論文設定） | 0.337/0.362/0.313 (+0.06) | 1.577/1.651/1.503 (+1.31) | 1.058/1.093/0.930 (+0.85) | 1.129/1.168/1.127 (+0.86) | 0.527/0.608/0.505 (+0.42) |
| B 該車種自己校正 | 同上 | 0.469/0.552/0.540 (+0.13) | 0.268/0.290/0.274 (−0.01) | 0.173/0.177/0.170 (+0.02) | （無校正集） |
| C 4 車種合併校正 | 0.831/0.792/0.782 (−0.75) | 0.785/0.855/0.798 (+0.54) | 0.444/0.419/0.431 (−0.01) | 0.367/0.372/0.411 (+0.31) | 0.516/0.404/0.441 (−0.41) |
| D 留一車種（未見過的車種） | 1.132/1.090/1.061 (−1.07) | 0.936/0.997/0.928 (+0.69) | 0.495/0.464/0.477 (−0.01) | 0.576/0.540/0.643 (+0.49) | 0.516/0.404/0.441 (−0.41) |

- **結論**：
  1. **校正依車種而定**。這與補充 Q 的 SUV 結果一致，且這次在 3 個新車種上都重現。
     - 用消防車校正套到其他車，一律高估 0.4–1.3 m，RMSE 0.5–1.6 m。
     - 用自己車種的校正，RMSE 為 0.17–0.47 m，與消防車本身的 0.34 m 同一水準。
  2. **跨車種泛化差**：
     - 合併校正或留一車種，RMSE 為 0.37–1.13 m。
     - 原因是不同車種的清晰度基準不同：車尾紋理量不同，f/22 下 L 圖的灰階清晰度分別為轎車 28.7、SUV 30.1、救護車 46.5。
     - 單一曲線無法同時涵蓋所有車種，需要依紋理正規化（M10 方向），或先分類車種再套用對應的校正。
  3. CA-DFDS 對比灰階：
     - A、B 兩種校正下 CA-DFDS 略優（約 0.01–0.08 m）。
     - C、D 兩種校正下灰階有時較好。
     - 只用 J 與論文權重相當。結論與前面相同：色彩權重的增益很小。
  4. 黑色仍是最難的顏色。例如轎車自己校正時，黑色 0.64 m、藍色 0.24 m。
- 檔案：
  - `results/multi_vehicle_carla.csv`（各設定、校正方式、車種、顏色）
  - `results/multi_vehicle_carla_by_distance.csv`
  - `results/multi_vehicle_carla_examples.jpg`（4.2 m 範例與 CARLA 遮罩）
  - `features/mv_*.npz`（含 `texture_f22`）
- 投影片格式的完整報告（各色逐距離表、校正曲線、3.7 m 前後誤差）：[docs/M7_MULTIVEHICLE_REPORT.md](docs/M7_MULTIVEHICLE_REPORT.md)

## issue #7｜實體相機實拍

- 未執行：需要實體相機，以及在已知距離拍攝模型車或實車，無法在實驗室主機上以程式完成。
- 拍攝完成後，可直接用 `cadfds_gpu.py`、`scripts/m4_auto_mask_*.py` 的流程處理。
