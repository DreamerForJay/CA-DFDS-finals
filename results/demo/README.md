# Demo 素材（網頁＋簡報）— 2026-11-18

依 [docs/DEMO_ASSETS.md](../../docs/DEMO_ASSETS.md) 產出。每個資料夾都有 `README.md`，寫明資料來源、設定和產生指令；影片中的數字都出自同一次推論的結果檔。

| 代號 | 資料夾 | 內容 | 影片 |
|---|---|---|---|
| C-A1 | [C-A1_teaser](C-A1_teaser/) | 三聯畫面 Teaser：輸入＋遮罩｜清晰度熱圖｜估測 vs 真實（紅色車 6.7 → 1.2 m） | 20 s／精華 13 s |
| C-B1 | [C-B1_camera_setup](C-B1_camera_setup/) | 相機配置示意圖（側視＋俯視）、三個距離的相機視角 | —（CARLA 第三人稱影片尚未產出） |
| C-C1 | [C-C1_input_output](C-C1_input_output/) | 近／中／遠 × 7 步（輸入 → 遮罩 → J／a_c／b_c → 清晰度 → σ → 距離），每步 PNG＋數值 JSON | 31 s／11 s |
| C-D1 | [C-D1_defocus_theory](C-D1_defocus_theory/) | 薄透鏡散焦曲線：對焦 15 m vs 7 m、f/1.4 vs f/22，含距離游標動畫 | 9 s／4.5 s |
| C-E1 | [C-E1_method_compare](C-E1_method_compare/) | 1×4 同步比較：Depth Pro（2024）、UniDepthV2（2025）、STMAP（1994）、CA-DFDS；逐幀誤差曲線；片尾 480 張 RMSE | 52 s／13.5 s |
| C-F1 | [C-F1_before_after](C-F1_before_after/) | 旋轉 180°（4 方法、480 張）；灰階 vs CIECAM02 清晰度拖曳對照 | 36 s／13 s |
| C-H1 | [C-H1_leaderboard](C-H1_leaderboard/) | 排行榜表（JSON／CSV／MD）：方法附年份與出處 | — |
| C-I1 | [C-I1_analysis](C-I1_analysis/) | 誤差 vs 距離、各車色、22 個車色資料夾；`charts.json` 給互動圖 | — |
| C-K1 | [C-K1_failure_cases](C-K1_failure_cases/) | 失敗案例圖＋一句原因（`captions.json`） | — |
| C-M1 | [C-M1_timing](C-M1_timing/) | CPU／GPU 處理時間圖 | — |

## 重點數字（全部可在各資料夾 `summary.json` 查到）

- 480 張整體 RMSE（手繪遮罩）：CA-DFDS 0.337 m、UniDepthV2 線性校正 0.357 m（原始 1.879）、**STMAP 0.478 m（新增，雙光圈）**、Depth Pro 線性校正 0.751 m（原始 0.871）。
- **旋轉 180°（新增）**：CA-DFDS 0.337 → 0.337 m（不變）；STMAP 0.478 → 0.785 m；UniDepthV2（線性校正）0.357 → 3.013 m；Depth Pro（線性校正）0.751 → 1.524 m。
- 影片片段（48 幀，四色 × 12 距離，shot01）：CA-DFDS 0.334、UniDepthV2 0.350、STMAP 0.450、Depth Pro 0.779 m。

## 產生方式

```bash
# 1. STMAP 基準（CPU）→ features/stmap_{cal,test}.npz；計時寫入 results/timing.json
python scripts/stmap_eval.py && python scripts/stmap_eval.py --timing
python scripts/build_calibrations.py && python scripts/compare_methods.py
# 2. 深度模型：旋轉 180° 推論＋demo 用深度圖（torch 環境，約 4 分鐘）
HF_HUB_OFFLINE=1 ../.venv/Scripts/python.exe baselines/demo_deep.py
# 3. 全部素材（約 5 分鐘；需 logs/bin/ffmpeg.exe 與微軟正黑體）
PATH="$PWD/logs/bin:$PATH" python scripts/make_demo_assets.py      # 或 --only C-A1 C-E1 ...
# 4. 網頁版（JPG 壓縮＋manifest.json）→ ../CA-DFDS/docs/assets/demo/
python scripts/export_web_assets.py
```

- 影片規格：1920×1080、30 fps、H.264（`-crf 23 -pix_fmt yuv420p -movflags +faststart`），每支 ≤ 7 MB。`video_short.mp4` 給簡報用。
- 網頁版在 [DreamerForJay/CA-DFDS](https://github.com/DreamerForJay/CA-DFDS) 的 `docs/assets/demo/`，`manifest.json` 列出每項素材的標題、說明、檔案與重點數字。
- 共同限制：全部是 CARLA 模擬；距離是資料集中 12 個分別拍攝的距離，不是連續影片；遮罩為資料集手繪遮罩（全自動遮罩結果見 RESULTS 補充 H）。
