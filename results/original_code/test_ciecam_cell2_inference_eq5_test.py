# 匯出自 test_ciecam.ipynb cell 2（原檔未修改）
import numpy as np
import cv2
import os
import glob
import csv

# 確保 numpy 相容性
if not hasattr(np, 'float_'):
    np.float_ = np.float64

try:
    import colour
except ImportError:
    print("錯誤: 找不到 'colour-science' 套件。")
    print("請先執行: pip install colour-science")
    exit()

# ================= 固定參數設定 =================
GROUND_TRUTH = [1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7, 5.2, 5.7, 6.2, 6.7]
W_J, W_AC, W_BC = 0.8136, 0.1782, 0.0081
PARAMS_POLY = [-6.041939e+00, 1.232324e+02, 75.828251]
OUTDIR = 'reverse_result/alternating_white&black&gray&blue/'

# 要輪流執行的顏色清單
# COLORS = ['red', 'green', 'blue', 'black', 'gray', 'deepGray', 'deep160Gray',  'deepRed', 'deepGreen', 'deepBlue','brown', 'chartreuse', 'cyan', 'magenta', 'orange', 'rose', 'springGreen', 'violet', 'yellow']
COLORS = ['red', 'green', 'blue', 'black']
# ================= 核心函式 =================

def poly2_model(x, coeffs):
    p = np.poly1d(coeffs)
    return p(x)

def compute_sharpness_ciecam02_acbc(img, mask, w_j=0.6, w_ac=0.2, w_bc=0.2):
    if img is None: return None
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_float = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
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

    lap_J = cv2.Laplacian(J, cv2.CV_64F, ksize=3)
    lap_ac = cv2.Laplacian(ac, cv2.CV_64F, ksize=3)
    lap_bc = cv2.Laplacian(bc, cv2.CV_64F, ksize=3)
    weighted_gradient = w_j * lap_J + w_ac * lap_ac + w_bc * lap_bc
    
    if mask is not None:
        if mask.shape[::-1] != (img.shape[1], img.shape[0]): 
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_pixels = weighted_gradient[mask > 0]
    else:
        valid_pixels = weighted_gradient.flatten()

    return np.std(valid_pixels) if len(valid_pixels) > 0 else None

# ================= 執行程序 =================

def run_color_process(color_name):
    # 動態定義路徑
    data_root_dir = rf'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas\reverse\firetruck_{color_name}_reverse\Test_dataset_Focus1500cm_f1.4_f22.0'
    # mask_root_dir = rf'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas\firetruck_{color_name}\Test_dataset_Focus1500cm_f1.4_f22.0_mask'
    mask_root_dir = rf'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas\reverse\Test_dataset_Focus1500cm_f1.4_f22.0_mask_reverse'
    output_dir = OUTDIR
    output_csv = os.path.join(output_dir, f'test_results_{color_name}.csv')

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*30} 開始處理顏色: {color_name.upper()} {'='*30}")
    print(f"資料目錄: {data_root_dir}")
    print(f"{'Folder':<8} | {'GT(m)':<8} | {'Sharpness':<12} | {'Pred(m)':<10} | {'Err(m)':<10} | {'Err %':<8}")
    print("-" * 100)

    mse_list = []
    csv_rows = []
    
    for i, gt_dist in enumerate(GROUND_TRUTH, 1):
        folder_name, mask_name = f"T{i:02d}", f"T{i}.npy"
        path_img_dir = os.path.join(data_root_dir, folder_name, 'R')
        path_mask = os.path.join(mask_root_dir, mask_name)

        if not os.path.exists(path_img_dir): 
            continue
        
        img_files = []
        for ext in ['*.jpg', '*.png', '*.bmp']:
            img_files.extend(glob.glob(os.path.join(path_img_dir, ext)))
        img_files.sort()

        mask = np.load(path_mask) if os.path.exists(path_mask) else None
        
        batch_scores = []
        for img_path in img_files:
            img = cv2.imread(img_path)
            score = compute_sharpness_ciecam02_acbc(img, mask, w_j=W_J, w_ac=W_AC, w_bc=W_BC)
            if score is not None: batch_scores.append(score)
        
        if batch_scores:
            avg_sharpness = np.mean(batch_scores)
            pred_m = poly2_model(avg_sharpness, PARAMS_POLY) / 100.0
            err = pred_m - gt_dist
            err_p = (abs(err) / gt_dist) * 100
            
            mse_list.append(err ** 2)
            
            # 即時列印輸出
            print(f"{folder_name:<8} | {gt_dist:<8.2f} | {avg_sharpness:<12.4f} | "
                  f"{pred_m:<10.2f} | {err:<10.2f} | {err_p:<7.2f}%")
            
            csv_rows.append([round(gt_dist, 2), round(pred_m, 2), round(err, 2), f"{round(err_p, 2)}%"])

    final_rmse = np.sqrt(np.mean(mse_list)) if mse_list else 0

    # 寫入 CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['實際距離(m)', '推測距離(m)', '誤差(m)', '誤差百分比'])
        writer.writerows(csv_rows)
        writer.writerow([]) 
        writer.writerow(['總體 RMSE', f"{round(final_rmse, 4)} m", '', ''])

    print("-" * 100)
    print(f"顏色 {color_name} 完成！存檔至: {output_csv}")
    print(f"該色總 RMSE: {final_rmse:.4f} m\n")

# ================= 主程式迴圈 =================

if __name__ == "__main__":
    for color in COLORS:
        run_color_process(color)
    print("所有顏色測試完成！")
