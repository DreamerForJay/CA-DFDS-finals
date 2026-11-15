# 匯出自 test_ciecam.ipynb cell 4（原檔未修改）
import numpy as np
import os
import cv2
import glob
from scipy.optimize import minimize
import colour

# 確保 numpy 類型相容
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# ================= 1. 特徵提取函式 =================
def extract_ciecam_features(img, mask):
    """提取 J, ac, bc 三個分量的 Laplacian 標準差"""
    if img is None: return None
    
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float64) / 255.0

    # CIECAM02 轉換
    XYZ = colour.sRGB_to_XYZ(img_float)
    XYZ_w = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']) * 100
    L_A, Y_b = 2000 / np.pi * 0.2, 20.0
    surround_cond = colour.VIEWING_CONDITIONS_CIECAM02['Average']
    
    try:
        spec = colour.XYZ_to_CIECAM02(XYZ * 100, XYZ_w, L_A, Y_b, surround=surround_cond)
        J, C, h = np.nan_to_num(spec.J), np.nan_to_num(spec.C), np.nan_to_num(spec.h)
        h_rad = np.deg2rad(h)
        ac, bc = C * np.cos(h_rad), C * np.sin(h_rad)
    except: return None

    # 分別計算三個通道的 Laplacian
    lap_J = cv2.Laplacian(J, cv2.CV_64F, ksize=3)
    lap_ac = cv2.Laplacian(ac, cv2.CV_64F, ksize=3)
    lap_bc = cv2.Laplacian(bc, cv2.CV_64F, ksize=3)

    if mask is not None:
        if mask.shape[::-1] != (img.shape[1], img.shape[0]):
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask > 0)
        return [np.std(lap_J[mask_bool]), np.std(lap_ac[mask_bool]), np.std(lap_bc[mask_bool])]
    else:
        return [np.std(lap_J), np.std(lap_ac), np.std(lap_bc)]

# ================= 2. 訓練資料預算階段 (包含檔名輸出) =================
def prepare_training_data(colors, base_path, mask_head):
    train_features = []  
    train_distances = [] 
    
    print("--- 開始特徵提取階段 (Preprocessing) ---")
    for color in colors:
        current_root = f"{base_path}_{color}"
        cal_head = os.path.join(current_root, "Test_dataset_Focus1500cm_f1.4_f22.0/")
        
        if not os.path.exists(cal_head):
            print(f"[跳過] 找不到顏色目錄: {current_root}")
            continue

        print(f"\n>>> 正在處理顏色資料夾: {color}")
        t_folders = sorted([d for d in os.listdir(cal_head) if d.startswith('T')], key=lambda x: int(x.replace('T', '')))

        for folder_name in t_folders:
            folder_num = int(folder_name.replace('T', ''))
            dist = 100 + (folder_num - 1) * 50 # 物理真實距離 (cm)
            
            mask_path = os.path.join(mask_head, f"T{folder_num}.npy")
            mask = np.load(mask_path) if os.path.exists(mask_path) else None

            img_dir = os.path.join(cal_head, folder_name, "R")
            img_files = glob.glob(os.path.join(img_dir, "*.*"))
            
            if not img_files:
                continue

            print(f"  [距離 {dist}cm] 目錄: {folder_name}")
            
            # 遍歷該資料夾內的所有圖片 (假設每個距離 10 張)
            folder_feats = []
            for img_path in img_files:
                # 輸出正在讀取的檔名
                print(f"    讀取圖片: {os.path.basename(img_path)}")
                
                img = cv2.imread(img_path)
                feat = extract_ciecam_features(img, mask)
                if feat:
                    folder_feats.append(feat)
            
            # 取該距離所有圖片特徵的平均值作為訓練點
            if folder_feats:
                avg_feat = np.mean(folder_feats, axis=0)
                train_features.append(avg_feat)
                train_distances.append(dist)
    
    return np.array(train_features), np.array(train_distances)

# ================= 3. 交替優化核心 =================
def alternating_optimization(X, y, max_iters=10):
    w = np.array([0.6, 0.2, 0.2])
    
    print("\n--- 開始交替優化 (Alternating Optimization) ---")
    for i in range(max_iters):
        # 階段 A: 固定權重，優化擬合曲線
        scores = np.dot(X, w)
        coeffs = np.polyfit(scores, y, 2)
        poly_model = np.poly1d(coeffs)
        
        # 階段 B: 固定曲線，優化權重
        def objective(weights):
            w_norm = weights / np.sum(weights)
            current_scores = np.dot(X, w_norm)
            predictions = poly_model(current_scores)
            return np.sqrt(np.mean((predictions - y)**2))

        res = minimize(objective, w, bounds=[(0.01, 1), (0.01, 1), (0.01, 1)])
        new_w = res.x / np.sum(res.x)

        if np.allclose(w, new_w, atol=1e-5):
            print(f"迭代 {i+1}: 已收斂。")
            break
        
        w = new_w
        print(f"迭代 {i+1}: 目前 RMSE = {res.fun:.4f} | 權重 J:{w[0]:.3f}, ac:{w[1]:.3f}, bc:{w[2]:.3f}")

    return w, coeffs

# ================= 4. 執行流程 =================
BASE_PATH = "./datas/training/firetruck"
MASK_PATH = r'D:\\Carla-0.10.0-Win64-Shipping\\Carla-0.10.0-Win64-Shipping\datas\\training\\Training_dataset_mask'
TRAIN_COLORS = ["white", "gray", "black"]

# 執行預算與優化
features, distances = prepare_training_data(TRAIN_COLORS, BASE_PATH, MASK_PATH)
best_w, best_coeffs = alternating_optimization(features, distances)

print("\n" + "="*50)
print("★ 優化流程結束 ★")
print(f"最終最佳權重 (Best Weights):")
print(f"  w_J  : {best_w[0]:.4f}")
print(f"  w_ac : {best_w[1]:.4f}")
print(f"  w_bc : {best_w[2]:.4f}")
print(f"擬合公式: Distance = {best_coeffs[0]:.4e}*x^2 + {best_coeffs[1]:.4e}*x + {best_coeffs[2]:.4f}")
print("="*50)
