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

## 任務 G｜現場 GUI 測試

- 指令：`python gui_demo.py --calibration results/calibration_opt.json`；截圖由 `logs/gui_screenshots.py` 自動操作同一個 GUI（代替檔案對話框選檔 → 載入遮罩 → 按「估測距離」→ 擷取視窗）
- 結果：

| 截圖 | 影像 | 清晰度 σ | 估測 | 真實 | 誤差 | 計算時間 |
|---|---|---|---|---|---|---|
| `results/gui_1.png` | 紅色車 T05 shot01 | 2.09 | 3.04 m | 3.20 m | −0.16 m | 3020 ms |
| `results/gui_2.png` | 藍色車 T09 shot01 | 5.55 | 5.68 m | 5.20 m | +0.48 m | 2115 ms |
| `results/gui_3.png` | 綠色車 T02 shot01 | 0.58 | 1.41 m | 1.70 m | −0.29 m | 2164 ms |

- 檔案：`results/gui_*.png`、`logs/gui_screenshots.py`、`logs/taskG.log`
- 備註：GUI 開啟、載入遮罩、估測、顯示真實距離與誤差皆正常；滑鼠塗抹遮罩功能未在本次截圖中測試
