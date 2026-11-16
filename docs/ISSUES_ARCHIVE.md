# Issue 封存（原 GitHub issue #1–#8）

> 2026-11-16 repo 重建時，把原 GitHub issue 移到這份文件（原 repo 刪除後，issue 連結會失效）。
> **之後的實驗待辦直接記在 [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md)，結果記在 [RESULTS.md](../RESULTS.md)，不再開 issue。**

| # | 標題 | 狀態 |
|---|---|---|
| 1 | [決賽前] RTX 5080 完整管線計時（CIECAM02 CPU／GPU、含遮罩） | ✅ 已完成 |
| 2 | [決賽前] 前期研究近距離比較表之量測硬體 | ⬜ 未完成 |
| 3 | [論文補強] 式 (5) 與式 (7) 清晰度定義對照 | ✅ 已完成 |
| 4 | [論文補強] 消融 E：校正集加入更多車色 | ✅ 已完成 |
| 5 | [論文補強] 自動車輛遮罩取代人工遮罩 | ✅ 已完成 |
| 6 | [論文補強] Gamma 統一後重測 19 色 | ✅ 已完成 |
| 7 | [論文補強] 實體相機實拍驗證 | ⬜ 未完成 |
| 8 | [論文補強] 多車種與失敗案例 | ⬜ 未完成 |

---

## #1 [決賽前] RTX 5080 完整管線計時（CIECAM02 CPU／GPU、含遮罩）

- 狀態：CLOSED（建立 2026-11-16，關閉 2026-11-16）

### 內容

**目的**：可行性頁目前只寫灰階 74.41 ms；最終 CIECAM02 版本 CPU 為 2263 ms／張，需 GPU 數據。

**硬體**：實驗室 RTX 5080 主機（Intel Core Ultra 7 265K、64 GB RAM），記錄 GPU 驅動、CUDA、Python 版本。

**步驟**
1. 分段計時：讀圖、遮罩（SAM 2 框提示或 YOLO-seg）、sRGB→CIECAM02、拉普拉斯＋標準差、校正曲線。
2. 版本：CPU（現況）、GPU（PyTorch／CuPy 實作 CIECAM02 正向轉換）、遮罩外框裁切後再轉換。
3. 30 張，暖機 3 張，報平均／中位數／FPS。

**輸出**：`results/timing_full_pipeline.json`
**期限**：11/18

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

RTX 5080 主機（Core Ultra 7 265K、64 GB），torch 2.11+cu128／CUDA 12.8、Python 3.12.7。紅色車 30 張（1.2–6.7 m），暖機 3 張不計。
| 版本 | sRGB→CIECAM02 | 總計（含讀圖、讀遮罩） | FPS |
|---|---|---|---|
| CPU 整張 | 2283.5 ms | 2408.1 ms | 0.42 |
| CPU 遮罩外框裁切 | 1284.7 ms | 1417.3 ms | 0.71 |
| GPU fp64 整張 | 17.8 ms | 92.9 ms | 10.8 |
| GPU fp32 整張 | 3.6 ms | 70.9 ms | 14.1 |

- 自動遮罩：YOLO11m 12.9 ms＋SAM ViT-B 106.9 ms（含 I/O 共 198.9 ms）。
- GPU 版為 `cadfds_gpu.py`，與 CPU 特徵相對誤差 1.8e-13，距離預測完全相同。
- 目前瓶頸是讀 PNG（約 50 ms）。
- 結果檔：`results/timing_full_pipeline.json`（RESULTS 補充 M）



---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）

---

## #2 [決賽前] 前期研究近距離比較表之量測硬體

- 狀態：OPEN（建立 2026-11-16）

### 內容

**目的**：簡報近距離表（STMAP 86.434、Depth Pro 1.273、UniDepthV2 0.353、CA-DFDS 1.187 秒／張）需標註量測硬體，目前資料中查不到。

**需要的資料**：學長 DFDS 前期研究（Chang et al., 2026）實驗紀錄、程式或論文中之 CPU／GPU 型號。

**輸出**：在本 issue 留言回覆硬體型號與出處（論文頁碼或檔案路徑）即可。

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

Table 3 的數字（86.434／1.273／0.353／1.187 s）出自 DFDS 論文（`paper/116.pdf` 第 4.4 節、`paper/DFDS.pdf` 第 18 頁）。

- 論文、簡報、`宇森_中文初稿.docx`、`paper/datacopy/*.md` 都**沒有記載 CPU／GPU 型號**；論文第 4.1 節只寫 Canon EOS-6D。
- 需向作者（Yu-Cheng Chen 或指導教授）確認。
- 在確認之前，簡報應註明「量測硬體未載明於原論文」。（RESULTS 補充 P）



---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）

---

## #3 [論文補強] 式 (5) 與式 (7) 清晰度定義對照

- 狀態：CLOSED（建立 2026-11-16，關閉 2026-11-16）

### 內容

**目的**：作品說明書式 (4) 與式 (7) 不一致（先加權再取標準差 vs 先取標準差再加權）。已確認推論用式 (5)、最佳化用式 (7)。

**步驟**：用 `features/*.npz`（共變異矩陣）同一組權重與校正集，計算兩種定義在 19 色測試距離之 RMSE。
**輸出**：`results/eq5_vs_eq7.csv`

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

同一組權重、同一校正集（白灰黑），19 色逐張 RMSE：

- 論文權重：式 (5) 0.337 m（R² 0.942），式 (7) 0.316 m（R² 0.923）。
  - 式 (7) 在紅色較好（0.194 vs 0.276），在藍、黑較差（0.446／0.575 vs 0.348／0.468）。
- J only 時兩式相同（0.311 m）。
- 建議作品說明書統一寫式 (5)（推論程式所用），並註明權重是用式 (7) 最佳化得到。
- 結果檔：`results/eq5_vs_eq7.csv`（RESULTS 補充 K）



---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）

---

## #4 [論文補強] 消融 E：校正集加入更多車色

- 狀態：CLOSED（建立 2026-11-16，關閉 2026-11-16）

### 內容

**目的**：補足消融表（A 灰階、B 固定權重、C 最佳化、D 只用 J 已完成，見 docs/NEXT_EXPERIMENTS.md M2）。

**步驟**：校正集用白＋灰＋黑＋紅＋綠＋藍（`cal_*.npz` 已抽取），重新最佳化權重，測其餘 13 個未參與校正之車色；不可用測試車色決定權重。
**輸出**：`results/ablation_E.csv`

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

- 校正集為白灰黑紅綠藍；權重只用校正集最佳化（SLSQP、Σw = 1），得 (0.9786, 0.0214, 0)。
- 測試集是未參與校正的 **14 色**（19 色扣掉紅綠藍黑灰；issue 寫 13 色）。
- 14 色平均：

  | 模型 | RMSE |
  |---|---|
  | E | 0.303 m |
  | J only | 0.301 m |
  | 論文權重 | 0.327 m |
  | 灰階 | 0.320 m |
  | 固定權重 (0.6, 0.2, 0.2) | 0.404 m |

  （A–D 在同一 6 色校正集上重算。）
- 權重仍集中於 J。
- 結果檔：`results/ablation_E.csv`、`results/ablation_E_weights.json`（RESULTS 補充 L）



---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）

---

## #5 [論文補強] 自動車輛遮罩取代人工遮罩

- 狀態：CLOSED（建立 2026-11-16，關閉 2026-11-16）

### 內容

**步驟**：SAM 2（框提示來自 YOLO）或 YOLO-seg 產生遮罩；記錄與人工遮罩 IoU、距離 RMSE 變化、遮罩耗時（RTX 5080）。
**輸出**：`results/auto_mask.csv`、範例圖 `results/auto_mask_examples/`

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

YOLO11m → SAM ViT-B（YOLO 沒偵測到時用近距離後備框），校正集也改用自動遮罩。

- IoU（與手繪遮罩比較）：測試 480 張平均 0.906（最低 0.660），校正 390 張平均 0.892。
- CA-DFDS RMSE：手繪 0.337 → 測試自動 0.310 → **校正、測試全自動 0.317 m**。
- 弱點：1.7–3.2 m 只能用後備框（2.2 m 時 0.207 → 0.327 m）。
- 時間：YOLO 約 13 ms、SAM 約 100–108 ms（RTX 5080）。
- 結果檔：`results/auto_mask.csv`、`results/auto_mask_summary.json`、`results/auto_mask_examples/`（RESULTS 補充 N）



---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）

---

## #6 [論文補強] Gamma 統一後重測 19 色

- 狀態：CLOSED（建立 2026-11-16，關閉 2026-11-16）

### 內容

**目的**：作品說明書註明部分深色／灰色影像為 gamma 1.0、其他為 2.2，影響車色比較公平性。
**步驟**：統一 gamma（全部 2.2 或全部 1.0）重產或重處理，跑 19 色評估。
**輸出**：`results/19colors_gamma_unified.csv`

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

- 檢查方式：比對各色同距離背景像素，共 50 組，log-log 斜率 0.844–1.057。
- **現有資料沒有 gamma 1.0／2.2 混用**（若混用，斜率應約為 2.2 或 0.45）；亮度差異來自 histogram 自動曝光（增益 0.68–1.40）。
- 全部轉成 gamma 1.0（線性光）重測 19 色：

  | 方法 | 現有資料 | gamma 1.0 |
  |---|---|---|
  | 灰階 | 0.332 m | 0.251 m |
  | CA-DFDS | 0.337 m | 0.276 m |
  | J only | 0.311 m | 0.267 m |

  此為由 8-bit 轉換，不是重新渲染，需用 CARLA 確認。
- 結果檔：`results/gamma_check.csv`、`results/19colors_gamma_unified.csv`（RESULTS 補充 O）



---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）

---

## #7 [論文補強] 實體相機實拍驗證

- 狀態：OPEN（建立 2026-11-16）

### 內容

**步驟**：可設定光圈與手動對焦之相機，對焦固定於測距區間之後；拍不同顏色模型車或實車於多個已知距離（每距離 ≥ 5 張）。
**回報**：相機與鏡頭型號、光圈、焦距、對焦距離、距離 RMSE、範例影像。
**輸出**：`results/real_capture/`

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

需要實體相機在已知距離實拍，無法在實驗室主機上用程式完成。



---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）

---

## #8 [論文補強] 多車種與失敗案例

- 狀態：OPEN（建立 2026-11-16）

### 內容

**步驟**
1. 轎車、SUV、貨車各至少 3 色，測 1–7 m 距離誤差。
2. 失敗案例：低紋理、曝光變化、強反光、超出 1–7 m；記錄遮罩面積與紋理量對誤差的影響。

**輸出**：`results/multi_vehicle.csv`、`results/failure_cases/`

---
**回報方式**：結果檔放 `results/`，`RESULTS.md` 新增一節（指令、資料路徑、硬體、數值表、檔案清單、問題），並在本 issue 留言貼結果摘要；完成後關閉 issue。數字只填實際量測值，未量測填「—」。

### 留言（DreamerForJay，2026-11-16）

- `datas/audi.etron`、`datas/ford.mustang` **其實都是同一台 SUV（nissan.patrol_2021）**：CARLA 0.10 沒有這兩個車種，`get_dataset2.py` 退回清單中的下一個車種。
- 套用消防車校正：SUV RMSE 1.27 m，平均高估 +1.10 m（1.7–5.7 m）→ 校正曲線依車種而定。
- 失敗案例：
  - 7.2 m（超出校正範圍）：0.48 m，低估。
  - 上下翻轉：與原圖完全相同。
  - 曝光 ×0.7／×1.4：0.49／0.62 m。
  - 紋理與誤差：Spearman ρ = −0.44（p = 0.058）。
  - 強反光：無資料，填「—」。
- 結果檔：`results/multi_vehicle.csv`、`results/failure_cases/`（RESULTS 補充 Q）


---
（實驗於實驗室 RTX 5080 主機執行；commit 05ad414，詳見 RESULTS.md「2026-11-16 GitHub issue 實驗」）
