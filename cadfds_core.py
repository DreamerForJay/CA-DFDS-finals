# -*- coding: utf-8 -*-
"""CA-DFDS 核心運算（與論文／學長校正程式 compute_sharpness_ciecam02_acbc 相同流程）。

清晰度（論文式 (4)(5)）：
  3x3 高斯 → sRGB→XYZ → CIECAM02（D65, L_A = 2000/π×0.2, Y_b = 20, Average）
  → J, a_c = C cos h, b_c = C sin h → 各通道 3x3 拉普拉斯 → 加權相加 → 遮罩內標準差 σ
為了讓任意權重都能快速重算，本模組對每張影像儲存三通道拉普拉斯響應在遮罩內的 3x3 共變異矩陣 Σ，
則 σ(w) = sqrt(wᵀ Σ w)（與 np.std 之母體標準差一致）。
"""
import os, re, glob, json
import numpy as np
import cv2

if not hasattr(np, 'float_'):
    np.float_ = np.float64

W_OPT = (0.8136, 0.1782, 0.0081)       # 論文最終權重（宇森 0516 最佳化結果）
W_FIXED = {'w1': (0.6, 0.2, 0.2), 'w2': (0.8, 0.1, 0.1)}


def imread(path):
    """支援中文路徑；讀檔失敗回傳 None。"""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_mask(path, shape=None):
    if path.lower().endswith('.npy'):
        m = np.load(path)
    else:
        m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    m = (m > 0).astype(np.uint8)
    if shape is not None and m.shape != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


def ciecam_channels(img_bgr):
    import colour
    blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    XYZ = colour.sRGB_to_XYZ(rgb)
    XYZ_w = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']) * 100
    spec = colour.XYZ_to_CIECAM02(XYZ * 100, XYZ_w, 2000 / np.pi * 0.2, 20.0,
                                  surround=colour.VIEWING_CONDITIONS_CIECAM02['Average'])
    J = np.nan_to_num(spec.J); C = np.nan_to_num(spec.C); h = np.deg2rad(np.nan_to_num(spec.h))
    return J, C * np.cos(h), C * np.sin(h)


def laplacian_cov(img_bgr, mask):
    """回傳遮罩內 (lap_J, lap_ac, lap_bc) 之 3x3 母體共變異矩陣。"""
    L = [cv2.Laplacian(x, cv2.CV_64F, ksize=3) for x in ciecam_channels(img_bgr)]
    V = np.stack([l[mask] for l in L])
    return np.cov(V, bias=True)


def gray_sigma(img_bgr, mask):
    """灰階基準：3x3 高斯 → 0.299R+0.587G+0.114B → 3x3 拉普拉斯 → 遮罩內標準差。"""
    g = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (3, 3), 0), cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F, ksize=3)[mask].std())


def sigma_eq5(cov, w):
    w = np.asarray(w, float)
    cov = np.asarray(cov)
    if cov.ndim == 2:
        return float(np.sqrt(w @ cov @ w))
    return np.sqrt(np.einsum('i,nij,j->n', w, cov, w))


def sharpness(img_bgr, mask, w=W_OPT):
    return sigma_eq5(laplacian_cov(img_bgr, mask), w)


def fit_quadratic(x, d_cm):
    co = np.polyfit(np.asarray(x, float), np.asarray(d_cm, float), 2)
    pred = np.polyval(co, x)
    r2 = 1 - ((d_cm - pred) ** 2).sum() / ((d_cm - np.mean(d_cm)) ** 2).sum()
    return co, float(r2)


DIST_RE = re.compile(r'D(\d+(?:\.\d+)?)_', re.I)


def discover(dataset_dir, aperture='R'):
    """dataset_dir 底下為 T01..Tnn/<aperture>/D<dist>_shotNN.png。回傳 [(path, T編號, 距離cm)]。"""
    out = []
    for tdir in sorted(glob.glob(os.path.join(dataset_dir, 'T*'))):
        m = re.search(r'T(\d+)$', os.path.basename(tdir))
        if not m:
            continue
        t = int(m.group(1))
        for f in sorted(glob.glob(os.path.join(tdir, aperture, '*.*'))):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            dm = DIST_RE.search(os.path.basename(f))
            out.append((f, t, float(dm.group(1)) * 100 if dm else np.nan))
    return out


def find_mask(mask_dirs, t):
    for d in mask_dirs:
        for name in (f'T{t}.npy', f'T{t:02d}.npy', f'T{t}.png', f'T{t:02d}.png'):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def save_calibration(path, w, coef, colors, r2):
    json.dump({'weights': list(map(float, w)), 'coef_cm': list(map(float, coef)),
               'formula': 'd_cm = c2*s^2 + c1*s + c0, s = sigma_eq5', 'calib_colors': colors, 'r2': r2},
              open(path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


def load_calibration(path):
    c = json.load(open(path, encoding='utf-8'))
    return np.array(c['weights']), np.array(c['coef_cm'])
