# -*- coding: utf-8 -*-
"""M5 / issue #6: gamma consistency of the dataset, and the effect of a gamma 1.0 / 2.2 difference on the 19-color RMSE.

Part 1 (check): for each color, compare its background (outside the dilated hand mask, lower image part) with a reference
color at the same distance. Exposure (histogram auto-exposure) is a gain -> log-log slope ~ 1; a gamma 1.0 vs 2.2 encoding
difference gives slope ~ 2.2 or ~ 0.45.  -> results/gamma_check.csv
Part 2 (sensitivity): re-encode every image as a gamma-1.0 (linear-light) 8-bit image, I' = round(255 * EOTF_sRGB(I/255)),
recompute features with the GPU pipeline (cadfds_gpu, fp64, identical to the CPU features to 1e-13), and evaluate:
  orig       : calibration and test as stored (current data)
  uniform_g1 : calibration and test both re-encoded to gamma 1.0
  mismatch   : calibration as stored, test re-encoded to gamma 1.0 (what a real gamma mix-up would do)
-> features/g1_*.npz, results/19colors_gamma_unified.csv
Usage (torch env): ../.venv/Scripts/python.exe scripts/m5_gamma.py
"""
import os, sys, glob, csv
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import cadfds_core as C
from evaluate import load, per_dist

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
TEST_MASK = os.path.join(DATA, 'firetruck_black', DS + '_mask')
CAL_MASK = os.path.join(DATA, 'training', 'Training_dataset_mask')
PAPER19 = ['red', 'green', 'blue', 'black', 'gray', 'deepGray', 'deep160Gray', 'deepRed', 'deepGreen', 'deepBlue',
           'brown', 'chartreuse', 'cyan', 'magenta', 'orange', 'rose', 'springGreen', 'violet', 'yellow']
v = np.arange(256) / 255.0
LUT_G1 = np.round(255 * np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)).astype(np.uint8)


def gray_img(p):
    return cv2.cvtColor(C.imread(p), cv2.COLOR_BGR2GRAY).astype(float)


def check():
    rows = []
    for cal, T, ref in ((False, 6, 'red'), (False, 12, 'red'), (True, 7, 'white'), (True, 13, 'white')):
        base = os.path.join(DATA, 'training') if cal else DATA
        m = C.load_mask(os.path.join(CAL_MASK if cal else TEST_MASK, f'T{T}.npy'))
        b = ~(cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0); b[:300] = False
        A = gray_img(sorted(glob.glob(os.path.join(base, f'firetruck_{ref}', DS, f'T{T:02d}', 'R', '*.png')))[0])
        for d in sorted(glob.glob(os.path.join(base, 'firetruck_*'))):
            f = sorted(glob.glob(os.path.join(d, DS, f'T{T:02d}', 'R', '*.png')))
            if not f:
                continue
            B = gray_img(f[0]); ok = b & (A > 15) & (A < 235) & (B > 15) & (B < 235)
            sl, ic = np.polyfit(np.log(A[ok]), np.log(B[ok]), 1)
            rows.append(dict(set='calib' if cal else 'test', T=T, reference=ref, color=os.path.basename(d)[10:],
                             loglog_slope=round(float(sl), 3), gain_at_100=round(float(np.exp(ic + (sl - 1) * np.log(100))), 3),
                             r=round(float(np.corrcoef(np.log(A[ok]), np.log(B[ok]))[0, 1]), 3), bg_pixels=int(ok.sum())))
    with open('results/gamma_check.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    s = np.array([r['loglog_slope'] for r in rows if r['color'] != r['reference']])
    print(f'gamma check: {len(s)} comparisons, slope range {s.min():.3f}-{s.max():.3f} (gamma 1.0 vs 2.2 would be ~2.2 or ~0.45)')


def extract(name, dataset, mask_dir, g):
    out = f'features/g1_{name}.npz'
    if os.path.exists(out):
        return
    items = C.discover(dataset); cov, gr, dist, ts, paths = [], [], [], [], []
    for p, t, d in items:
        mp = C.find_mask([mask_dir], t)
        if mp is None:
            continue
        img = cv2.LUT(C.imread(p), LUT_G1); m = C.load_mask(mp, img.shape)
        ci, cm = C.crop_to_mask(img, m)
        cov.append(g.cov_from_bgr(ci, cm)); gr.append(C.gray_sigma(ci, cm)); dist.append(d); ts.append(t); paths.append(p)
    np.savez(out, cov=np.array(cov), gray=np.array(gr), dist_cm=np.array(dist), t=np.array(ts), path=np.array(paths))
    print('saved', out, len(cov), flush=True)


def feat(D, s):
    return D['gray'] if isinstance(s, str) else C.sigma_eq5(D['cov'], s)


def curve(cal, s):
    X, Y = [], []
    for D in cal:
        x, d = per_dist(feat(D, s), D['dist']); X += list(x); Y += list(d)
    return C.fit_quadratic(np.array(X), np.array(Y))


def rmse(co, D, s):
    x = feat(D, s); ok = np.isfinite(x) & np.isfinite(D['dist'])
    return float(np.sqrt(np.mean((np.polyval(co, x[ok]) - D['dist'][ok]) ** 2)) / 100)


if __name__ == '__main__':
    check()
    import cadfds_gpu as G
    g = G.CIECAM02GPU()
    for c in ('white', 'gray', 'black'):
        extract(f'cal_{c}', os.path.join(DATA, 'training', f'firetruck_{c}', DS), CAL_MASK, g)
    for c in PAPER19:
        extract(f'test_{c}', os.path.join(DATA, f'firetruck_{c}', DS), TEST_MASK, g)
    cal_o = [load(f'features/cal_{c}.npz') for c in ('white', 'gray', 'black')]
    cal_g = [load(f'features/g1_cal_{c}.npz') for c in ('white', 'gray', 'black')]
    te_o = {c: load(f'features/test_{c}.npz') for c in PAPER19}; te_g = {c: load(f'features/g1_test_{c}.npz') for c in PAPER19}
    settings = {'gray': 'gray', 'opt(0.8136,0.1782,0.0081)': C.W_OPT, 'Jonly(1,0,0)': (1.0, 0.0, 0.0)}
    rows = []
    for sn, s in settings.items():
        for proto, cal, te in (('orig', cal_o, te_o), ('uniform_g1', cal_g, te_g), ('mismatch_cal2.2_test1.0', cal_o, te_g)):
            co, r2 = curve(cal, s); per = {c: rmse(co, te[c], s) for c in PAPER19}
            r = dict(setting=sn, protocol=proto, calib_r2=round(r2, 4), mean19=round(float(np.mean(list(per.values()))), 4))
            r.update({c: round(v, 4) for c, v in per.items()}); rows.append(r)
            print(f'{sn:28s} {proto:24s} R2 {r2:.4f} mean19 {r["mean19"]:.3f}', flush=True)
    with open('results/19colors_gamma_unified.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('saved results/19colors_gamma_unified.csv')
