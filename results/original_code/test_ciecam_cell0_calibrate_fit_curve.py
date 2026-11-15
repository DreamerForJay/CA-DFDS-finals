# 匯出自 test_ciecam.ipynb cell 0（原檔未修改）
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit

# 確保 numpy 有 float_ 定義
if not hasattr(np, 'float_'):
    np.float_ = np.float64

try:
    import colour
except ImportError:
    print("錯誤: 找不到 'colour-science' 套件。")
    exit()

# ================= 參數設定 =================
# 使用萬用字元匹配所有 firetruck 系列資料夾
COLORS = ["white", 'black', 'gray', 'blue']  # 可以調參數，看要抓甚麼顏色

# 根據你的資料夾結構組合前綴
# 假設路徑是 ./datas/firetruck_red/Training_dataset...
DATA_BASE_PATH = "./datas/training/firetruck" # 自行調整資料夾參數

LARGER = "/R" 
BASE = 100   
PARA = 50    
MIN_VALID_DIST = 50 

W_J, W_AC, W_BC = 0.8136, 0.1782, 0.0081 # 目前學長跑最佳化出來的

# ================= 核心函式 (J, ac, bc) =================
def compute_sharpness_ciecam02_acbc(img, mask, w_j=0.6, w_ac=0.2, w_bc=0.2):
    if img is None: return None
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float64) / 255.0

    XYZ = colour.sRGB_to_XYZ(img_float)
    XYZ_w = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']) * 100
    L_A = 2000 / np.pi * 0.2
    Y_b = 20.0
    surround_cond = colour.VIEWING_CONDITIONS_CIECAM02['Average']
    
    try:
        specification = colour.XYZ_to_CIECAM02(
            XYZ * 100, XYZ_w, L_A, Y_b, surround=surround_cond
        )
        J = np.nan_to_num(specification.J)
        C = np.nan_to_num(specification.C)
        h = np.nan_to_num(specification.h)
        h_rad = np.deg2rad(h)
        ac = C * np.cos(h_rad)
        bc = C * np.sin(h_rad)
    except Exception:
        return None

    lap_J = cv2.Laplacian(J, cv2.CV_64F, ksize=3)
    lap_ac = cv2.Laplacian(ac, cv2.CV_64F, ksize=3)
    lap_bc = cv2.Laplacian(bc, cv2.CV_64F, ksize=3)

    weighted_gradient = w_j * (lap_J) + w_ac * (lap_ac) + w_bc * (lap_bc)
    
    if mask is not None:
        if mask.shape[::-1] != (img.shape[1], img.shape[0]): 
             mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask > 0)
        valid_pixels = weighted_gradient[mask_bool]
    else:
        valid_pixels = weighted_gradient.flatten()

    return np.std(valid_pixels) if len(valid_pixels) > 0 else None

# ================= 主程式 =================
def train_color_list_mode():
    all_dist_list = []
    all_score_list = []

    # --- 第一層迴圈：遍歷顏色陣列 ---
    for color in COLORS:
        # 動態組合資料夾路徑
        current_root = f"{DATA_BASE_PATH}_{color}"
        
        cal_head = os.path.join(current_root, "Test_dataset_Focus1500cm_f1.4_f22.0/")
        mask_head = 'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas\\training\Training_dataset_mask'

        if not os.path.exists(cal_head):
            print(f"跳過：路徑不存在 {cal_head}")
            continue

        print(f"\n>>> 正在處理顏色: {color.upper()}")

        # 取得 T1, T2, T3... 資料夾
        all_subs = [d for d in os.listdir(cal_head) if os.path.isdir(os.path.join(cal_head, d))]
        t_folders = sorted([d for d in all_subs if d.startswith('T')], key=lambda x: int(x.replace('T', '')))

        # --- 第二層迴圈：遍歷各距離資料夾 ---
        for folder_name in t_folders:
            folder_num = int(folder_name.replace('T', ''))
            current_dist = BASE + (folder_num - 1) * PARA
            
            # 讀取 Mask (npy)
            mask_path = os.path.join(mask_head, f"T{folder_num}.npy")
            mask = np.load(mask_path) if os.path.exists(mask_path) else None

            # 讀取圖片路徑
            img_dir = os.path.join(cal_head, folder_name, LARGER.strip("/"))
            img_files = [f for f in glob.glob(os.path.join(img_dir, "*.*")) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
            
            if not img_files: continue

            local_scores = []
            for f in img_files:
                img = cv2.imread(f)
                val = compute_sharpness_ciecam02_acbc(img, mask, w_j=W_J, w_ac=W_AC, w_bc=W_BC)
                if val is not None: local_scores.append(val)
            
            if local_scores:
                mean_val = np.mean(local_scores)
                all_score_list.append(mean_val)
                all_dist_list.append(current_dist)
                print(f"  [{color}] Dist: {current_dist}cm | Score: {mean_val:.4f}")

    # --- 數據擬合與繪圖 (與之前邏輯相同) ---
    X = np.array(all_score_list)
    Y = np.array(all_dist_list)
    
    # 過濾近距離干擾
    valid_mask = Y >= MIN_VALID_DIST
    X_train, Y_train = X[valid_mask], Y[valid_mask]

    if len(X_train) < 3:
        print("有效數據不足！")
        return

    # 多項式擬合
    coeffs = np.polyfit(X_train, Y_train, 2)
    poly_model = np.poly1d(coeffs)

    # 繪製結果
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, c='blue', alpha=0.5, label='All Samples')
    x_range = np.linspace(X.min(), X.max(), 100)
    plt.plot(x_range, poly_model(x_range), 'r-', label='Combined Fit')
    plt.title(f"Calibration for Colors: {', '.join(COLORS)}")
    plt.xlabel("Sharpness Score")
    plt.ylabel("Distance (cm)")
    plt.legend()
    plt.grid(True)
    plt.show()
    # --- 數據擬合 ---
    X = np.array(all_score_list)
    Y = np.array(all_dist_list)

    # 1. 篩選數據 (排除小於 MIN_VALID_DIST 的點)
    valid_mask = Y >= MIN_VALID_DIST
    X_train, Y_train = X[valid_mask], Y[valid_mask]

    if len(X_train) < 3:
        print("\n[錯誤] 有效數據點不足，無法進行二次項擬合。")
        return

    # 2. 進行二次多項式擬合: Dist = a*Score^2 + b*Score + c
    coeffs = np.polyfit(X_train, Y_train, 2)
    a, b, c = coeffs  # 解構係數
    poly_model = np.poly1d(coeffs)

    # 3. 計算決定係數 R^2 (評估準確度)
    y_pred = poly_model(X_train)
    r_squared = 1 - (np.sum((Y_train - y_pred)**2) / np.sum((Y_train - np.mean(Y_train))**2))

    # ================= 輸出參數 =================
    print("\n" + "="*50)
    print("  ★ 擬合參數輸出 (Final Calibration Parameters) ★")
    print("="*50)
    print(f"  二次項 (a): {a:.10f}")
    print(f"  一次項 (b): {b:.10f}")
    print(f"  常數項 (c): {c:.10f}")
    print(f"  準確度 (R²): {r_squared:.4f}")
    print("-"*50)
    print(f"  公式: Distance = ({a:.6e})*x² + ({b:.6e})*x + ({c:.6f})")
    print("="*50)

if __name__ == "__main__":
    train_color_list_mode()
