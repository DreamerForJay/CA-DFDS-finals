# 如何打開 GUI Demo 畫面

`gui_demo.py` 是現場展示用的 GUI，會根據車輛影像和遮罩估測車距。截圖範例：`results/gui_1.png` 到 `gui_3.png`。

## 1. 啟動

**方法 A：直接點兩下（實驗室主機）**

到 repo 資料夾 `D:\Carla-0.10.0-Win64-Shipping\CA-DFDS-finals`，點兩下 **`run_gui.bat`**。

**方法 B：在終端機啟動**

```bat
cd /d D:\Carla-0.10.0-Win64-Shipping\CA-DFDS-finals
.venv-cadfds\Scripts\python.exe gui_demo.py --calibration results\calibration_opt.json
```

- 必須使用 `.venv-cadfds` 這個環境。`venv-carla` 裡的 `colour` 是同名的錯誤套件，不能用。
- 不加 `--calibration`，或找不到檔案時，會改用論文的校正曲線。

## 2. 操作步驟

1. 按「**開啟影像**」，選一張 f/1.4 的影像（`R` 資料夾）。範例：
   `<DATA>\firetruck_red\Test_dataset_Focus1500cm_f1.4_f22.0\T01\R\D1.2_shot01.png`
2. 按「**載入遮罩**」，選相同距離編號的遮罩。範例：
   `<DATA>\firetruck_black\Test_dataset_Focus1500cm_f1.4_f22.0_mask\T1.npy`
   - 各顏色共用這份遮罩。
   - `T01` 對應 `T1.npy`，`T12` 對應 `T12.npy`；T13（7.2 m）沒有遮罩。
   - 也可以不載入遮罩，直接用滑鼠左鍵在車上塗抹。右鍵或「清除遮罩」可以清除，「筆刷」調整塗抹大小。
3. 按「**估測距離**」，右側會顯示清晰度 σ 和估測距離。檔名有 `D<距離>_` 時，也會顯示真實距離和誤差。
   - 每張約需 2–3 秒，因為 CIECAM02 在 CPU 上算一張約 2.3 秒。

`<DATA>` = `D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas`

## 3. 自動截圖（不需手動操作）

```bash
.venv-cadfds/Scripts/python.exe logs/gui_screenshots.py results <影像> <遮罩> [<影像> <遮罩> ...]
```

## 4. 在別台電腦上執行

1. `pip install -r requirements.txt`（需要 numpy、opencv-python、pillow、colour-science）。
2. 執行 `python gui_demo.py`。tkinter 是 Python 內建的，不需要另外安裝。
3. 影像資料不在 git 中，需要另外取得。
