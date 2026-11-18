# C-D1 散焦理論曲線（對焦 7 m vs 15 m；f/1.4 vs f/22）

- 薄透鏡模糊圈直徑 c = f²/N · |d − s| / (d·(s − f))，換算成像素（像素間距 36 mm / 2048）。f = 40 mm，N = 1.4 或 22，s = 15 m 或 7 m。
- `curves.csv`（0.5–20 m，每 0.05 m）可直接給網頁做互動曲線；`per_frame.csv` 為 1–7 m 每 0.1 m。
- `summary.json` 另附校正集實測的 CA-DFDS σ（每距離平均），可與理論曲線並列：理論模糊量越大 → 實測 σ 越小。
- 重點：對焦在 15 m 時，1–7 m 內模糊量隨距離單調變化，不會有「對焦點前後模糊量相同」的歧義；f/22 幾乎不模糊（STMAP 用它當參考影像）。
- 這是理論值，不是 CARLA 量測；CARLA 的景深是渲染器近似。
- 影片：距離游標 7 → 1 m，三種設定的模糊圈同步變化（`video.mp4` 9.0 s，`video_short.mp4` 4.5 s）。
- `aperture_pairs/red_<距離>_f1.4.jpg`／`_f22.jpg`：同一幀 f/1.4（R 資料夾）與 f/22（L 資料夾）實際影像，同裁切、同尺寸（1600×1200），可做拖曳對照（DEMO_PLAN M3）。
- 這份資料同時回應 CA-DFDS 網站 TODO「散焦理論曲線的相機參數」。
- 產生：`python scripts/make_demo_assets.py --only C-D1`
