# CA-DFDS 決賽補充實驗（實驗室電腦執行用）

> ⚠ 目前最優先：Demo 影片／圖表素材需求見 [`docs/DEMO_ASSETS.md`](docs/DEMO_ASSETS.md)。

> **2026-11-15 更新：下一階段實驗清單（審查建議整理、依優先順序）見 [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md)，請優先執行。**


> 📌 **接手者（AI／同學）請先讀 [AGENTS.md](AGENTS.md)**：目前進度、協作規則、Git 身分、Google Drive 同步方式。變更紀錄見 [docs/CHANGELOG.md](docs/CHANGELOG.md)。
> 2026-11-15：任務 A–G 已全部完成，結果見 [RESULTS.md](RESULTS.md) 與 [docs/LAB_REPORT_2026-11-15.md](docs/LAB_REPORT_2026-11-15.md)。
> 🖥️ **開啟 GUI demo**：
>   - **新版 v3**：點兩下 `run_gui_new.bat` 開啟本機展示（http://127.0.0.1:8765/），支援四方法同步比較、CPU／CUDA 重跑與實驗影片。執行 `run_gui_new.bat --desktop` 可開啟原桌面介面，操作自動遮罩、自選影像與影片。
>   - **學長舊版**：點兩下 `run_gui_old_senior.bat`（`gui_old_senior.py`）。只有論文權重單一方法。
>   - 說明見 [docs/GUI_DEMO.md](docs/GUI_DEMO.md)。
> 🧭 **整體架構、遮罩（SAM 或 YOLO）、與其他方法比較**：見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

> 給實驗室電腦上的 AI／同學：請依本文件「任務清單」依序執行，完成後依〈回傳清單〉把 `results/` 打包傳回。
> 本 repo 只包含程式與說明，**不含影像資料**；影像與遮罩在實驗室電腦（RTX 5080 主機）上。

CA-DFDS（Color-Aware Depth From Defocus with Segmentation）：單一相機、f/1.4、對焦 15 m（後對焦），
於車輛遮罩內計算 CIECAM02 感知清晰度，經二次校正曲線估測 1–7 m 車距。

---

## 0. 環境

```bash
conda create -n cadfds python=3.10 -y && conda activate cadfds
pip install -r requirements.txt
# 影片轉檔需要 ffmpeg（Windows: winget install Gyan.FFmpeg；Ubuntu: sudo apt install ffmpeg）
```

自我檢查（CARLA 校正資料任一色即可）：

```bash
python -c "import cadfds_core, colour, cv2; print('ok')"
```

## 1. 資料位置（請先找到並填入）

資料夾結構與學長的程式相同：

```text
<DATA>/firetruck_<color>/<資料集名稱>/T01..Tnn/R/D<距離m>_shotNN.png    # R = f/1.4
<DATA>/training/Training_dataset_mask/T1.npy ...                       # 校正距離 1.0–7.0 m 之遮罩
<DATA>/firetruck_<color>/<資料集名稱>_mask/T<n>.npy                     # 測試集之遮罩（學長 mask 工具輸出位置）
```

| 用途 | 距離 | 資料集（請確認實際名稱） |
|---|---|---|
| 校正集（白、灰、黑） | 1.0–7.0 m，每 0.5 m，13 組 × 10 張 | `Test_dataset_Focus1500cm_f1.4_f22.0`（training 底下） |
| **測試集（論文表 7、9、10 之數據來源）** | 1.2–6.7 m，12 組 | 學長電腦上 1.2–6.7 m 之資料夾（`get_dataset2.py` 中 `DFD_DISTANCES` 改為測試距離所產生） |

> 「未參與校正」：校正曲線只用白、灰、黑三色車輛擬合；紅、綠、藍等測試車色從未用於擬合，用來檢驗對新車色的泛化。

距離由檔名 `D<距離>_` 解析；遮罩依 `T<n>` 對應，`--mask_dir` 可給多個資料夾依序搜尋。

---

## 2. 任務清單（依優先順序）

### 任務 A｜抽取特徵（其餘任務都需要）

```bash
# 校正集（白、灰、黑）
for c in white gray black; do
  python extract_features.py --dataset <DATA>/training/firetruck_$c/Test_dataset_Focus1500cm_f1.4_f22.0 \
      --mask_dir <DATA>/training/Training_dataset_mask --name cal_$c --workers 8
done
# 測試集（1.2–6.7 m）：至少紅、綠、藍、黑；若 19 色都在，請全部跑
for c in red green blue black; do
  python extract_features.py --dataset <DATA>/firetruck_$c/<測試資料集> \
      --mask_dir <DATA>/firetruck_$c/<測試資料集>_mask <DATA>/<共用測試遮罩資料夾> --name test_$c --workers 8
done
```

### 任務 B｜表 8 於「獨立測試距離」重做（最重要）

論文表 8 目前在校正距離評估；請在 1.2–6.7 m 測試集重做：

```bash
python evaluate.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz \
  --test features/test_red.npz features/test_green.npz features/test_blue.npz features/test_black.npz \
  --extra "reopt5=0.9532,0.0468,0" "Jonly=1,0,0" --out results/table8_testdist.csv
python reoptimize.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz --form 5 --out results/reopt_eq5.json
python reoptimize.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz --form 7 --out results/reopt_eq7.json
```

同時會輸出 `results/calibration_opt.json`（最終校正曲線，GUI 與 demo 使用）。

### 任務 C｜決賽 demo 影片（測試集，車輛由遠而近）

```bash
python make_demo_video.py --calibration results/calibration_opt.json \
  --features features/test_red.npz features/test_green.npz features/test_blue.npz \
  --labels 紅色車 綠色車 藍色車 --mask_dir <測試遮罩資料夾…> --out results/CA-DFDS_demo_testdist.mp4
# 想要更像連續影片：加 --shots all
```

Linux 若無微軟正黑體：`--font /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc`。

### 任務 D｜處理時間（必須在原本量 74.41 ms 的同一台 RTX 5080 主機）

```bash
python timing.py --dataset <任一測試資料集> --mask_dir <遮罩> --n 30 --out results/timing.json
```

### 任務 E｜19 種車色 × 最佳化權重（若 19 色測試資料在電腦上）

對每一色執行任務 A 的測試集抽取，再：

```bash
python evaluate.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz \
  --test features/test_*.npz --out results/19colors_opt.csv
```

### 任務 F｜找學長原始程式（請把找到的檔案複製到 `results/original_code/`）

1. 權重交替最佳化程式（`scipy.optimize.minimize`、SLSQP）：記錄**權重限制（是否非負、總和為 1）**、**停止條件、最大迭代次數**、目標函數是逐張或逐距離平均。
2. 測試／推論程式：確認最終推論是否用「先加權再取標準差」（式 5）。
3. 近距離版本 GUI（作品說明書圖 5 之介面）：原始碼與執行方式。

### 任務 G｜現場 GUI 測試

```bash
python gui_old_senior.py --calibration results/calibration_opt.json   # 學長舊版（任務 G 當時的版本）；新版為 gui_new_v3.py
```

開啟測試影像 → 載入遮罩（或用滑鼠塗抹車輛）→「估測距離」。請截圖 2–3 張存到 `results/gui_*.png`（已完成：`results/gui_old_senior_*.png`）。

---

## 3. 回傳清單

將下列檔案壓縮成 `cadfds_results_<日期>.zip` 傳回：

- `results/table8_testdist.csv`、`results/reopt_eq5.json`、`results/reopt_eq7.json`、`results/calibration_opt.json`
- `results/CA-DFDS_demo_testdist.mp4`
- `results/timing.json`
- `results/19colors_opt.csv`（若有）
- `results/original_code/`（若有找到）
- `results/gui_*.png`
- `features/*.npz`（小檔，方便本機重算）
- 一段文字紀錄：資料集實際路徑、遇到的錯誤與處理方式

## 4. 檔案說明

| 檔案 | 說明 |
|---|---|
| `cadfds_core.py` | 清晰度、灰階基準、資料搜尋、校正曲線 |
| `extract_features.py` | 每張影像 → 3x3 共變異矩陣（任意權重可即時重算） |
| `evaluate.py` | 校正 + 評估（逐距離／逐張兩種協定），輸出 CSV 與校正曲線 |
| `reoptimize.py` | 明確設定之權重交替最佳化 |
| `make_demo_video.py` | demo 影片（H.264） |
| `timing.py` | 灰階與 CIECAM02 版本處理時間 |
| `gui_new_v3.py` + `run_gui_new.bat` | **新版 v3** GUI：6 種方法比較、全自動遮罩、影片 |
| `gui_old_senior.py` + `run_gui_old_senior.bat` | **學長舊版** GUI：論文權重單一方法（保留對照） |
| `baselines/` | SAM／UniDepth／Depth Pro（GPU）背景程式與批次評估 |
| `docs/IMPROVEMENT_PLAN.md` | 改善計畫 |

---

## 5. 完成後：同步回 GitHub（必做）

每完成一項任務就同步一次，讓本機（決賽簡報製作端）能直接取用：

```bash
git pull
mkdir -p results
# 1) 結果檔放在 results/；單檔 > 95 MB 的影片請改傳雲端並在 RESULTS.md 寫連結
git add results/ -f
# 2) 更新 RESULTS.md：每項任務一節，寫「指令、資料路徑、數值結果（表格）、檔案清單、遇到的問題」
# 3) 更新 docs/IMPROVEMENT_PLAN.md：完成的項目把 🔄/⬜ 改為 ✅，並在該列補一句結果
git add RESULTS.md docs/IMPROVEMENT_PLAN.md
git commit -m "lab results: <任務代號與一句摘要>"
git push
```

`RESULTS.md` 範本：

```markdown
## 任務 B｜表 8（測試距離 1.2–6.7 m）
- 執行日期／電腦：
- 指令：
- 資料路徑：
- 結果：（貼 CSV 表格）
- 檔案：results/table8_testdist.csv
- 備註：
```
