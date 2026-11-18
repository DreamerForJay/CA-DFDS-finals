# CA-DFDS Demo 與方法比較展示規劃（2026-11-18）

參考學術專案頁的呈現方式：Teaser 影片 → 方法總覽 → 輸入／輸出 Demo → 互動對照 → 與其他方法同畫面比較 → 失敗案例。
目標：評審一眼看懂「一張後對焦影像 → 車輛遮罩 → 清晰度 σ → 距離 d」，並看出與單眼深度學習方法的差異。

> 規則：所有數字須來自實驗輸出檔（`results/`）；未量測者標「—」，不得以估計值代替。

## 一、展示項目

| # | 項目 | 內容 | 現有素材 | 需實驗室產出 |
|---|---|---|---|---|
| D1 | Teaser 影片（≤30 s） | 車輛由遠而近，畫面疊上遮罩與即時距離讀數及 GT | `results/CA-DFDS_demo_testdist.mp4` | 加 GT 距離與誤差顯示的版本 |
| D2 | 輸入→輸出 Demo | 影像 → SAM 2（2024）遮罩 → CIECAM02 J／a_c／b_c → 拉普拉斯清晰度 → σ → d，逐步出現 | 海報系統架構 (a)–(e) 圖 | 3 張代表距離（近／中／遠）的各階段中間圖與數值（JSON） |
| D3 | 光圈／對焦互動對照 | 滑桿切換 f/1.4 與 f/22、前對焦與後對焦（15 m）同場景，下方顯示模糊量曲線 | 理論散焦曲線圖 | 同場景兩種光圈、兩種對焦距離的成對影像 |
| D4 | 與其他方法同畫面比較 | 同一輸入，四欄：Depth Pro（2024）、UniDepthV2（2025）、STMAP（1994）、CA-DFDS；各欄顯示估計距離與誤差 | `results/method_comparison.md`（數據） | 各方法對 13 個距離的輸出圖與數值；串成依距離播放的影片 |
| D5 | 旋轉 180° 穩健性 | 同一影像原始 vs. 旋轉 180° 的估計結果並排 | 近距離實拍數據（報告書表 5） | 實拍影像兩種方向的輸出圖 |
| D6 | 多車色 | 紅／綠／藍／黑／白等車色同距離並排與誤差 | `results/19colors_*.csv` | 每色一張輸出圖 |
| D7 | GUI 操作錄影 | `gui_new_v3.py` 載入影像 → 自動遮罩 → 輸出距離，全程錄影 | `results/gui_new_v3_*.png` | 60 s 內螢幕錄影 |
| D8 | 失敗案例 | 遮罩錯誤、強逆光、超出校正範圍等 | `results/failure_cases/` | 每例一句原因 |

## 二、實驗室工作清單（依優先順序）
1. **D4 比較**：13 個距離 × 4 方法輸出圖，存 `results/demo/compare/<dist>_<method>.png`，數值存 `results/demo/compare.csv`（需與 `method_comparison.md` 一致）。
2. **D1 Teaser**：`make_demo_video.py` 加 GT 與誤差欄，輸出 `results/demo/teaser.mp4`。
3. **D2 中間輸出**：近／中／遠各一張，輸出遮罩、J／a_c／b_c、清晰度圖與 σ、d 數值。
4. **D7 GUI 錄影**：OBS 錄 1080p，≤60 s。
5. **D3、D5、D6** 成對影像與輸出圖。
6. **D8 失敗案例**整理。

## 三、檔案規格
- 影片：1920×1080、30 fps、H.264（`-crf 23 -pix_fmt yuv420p`），單檔 ≤ 25 MB；附首幀 JPG。
- 對照圖：同組 PNG 尺寸完全相同，寬 1600 px。
- 命名：`<demo編號>_<內容>_<日期>`，例如 `D4_method_compare_1120.mp4`。
- 上傳：本 repo `results/demo/`；網頁壓縮版放 `CA-DFDS`（GitHub Pages）`docs/assets/`；原始大檔放雲端並記錄於 `docs/MATERIALS_NEEDED.md`。

## 四、網頁區塊（GitHub Pages：CA-DFDS/docs/index.html）
1. 標題、作者、論文／影片／程式碼按鈕
2. D1 Teaser
3. 方法總覽（系統架構 (a)–(e)）＋ D2 步驟展示
4. D3 光圈／對焦互動滑桿
5. D4 方法比較（方法名附年份）＋ D5 旋轉穩健性
6. D6 多車色、D7 GUI 錄影
7. D8 失敗案例與限制
8. 引用格式

## 五、簡報對應
- 決賽簡報可加一頁「Demo 展示」：D1 Teaser＋D4 比較截圖＋QR code 連到 Demo 頁。
- 素材未完成前，簡報與海報只放已存在的素材。
