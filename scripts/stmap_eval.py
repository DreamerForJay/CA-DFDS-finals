# -*- coding: utf-8 -*-
"""STMAP baseline (Subbarao & Surya 1994, spatial-domain DFD with two apertures) → features/stmap_{cal,test}.npz

Same σ formula as the senior's stmap_dfds_rotation_eval.py (eq. 1 of the DFDS paper):
  G = sqrt(16·Σ(g1−g2)² / Σ(∇²g1)²),  σ = |G / (A²−1)|^¼,  A = 22/1.4
  g1 = f/1.4 image (R folder), g2 = f/22 image (L folder), grayscale.
STMAP has no segmentation: σ is computed over the whole bounding box of the hand-drawn mask (no pixel masking).
Rotation 180° (paper Table 4 protocol, same as the senior's script): the image is rotated but the sampling box stays
where it was, because STMAP cannot re-locate the vehicle.
Note: STMAP needs TWO images (two apertures); CA-DFDS needs only the f/1.4 image.

Calibration: same calibration set and protocol as CA-DFDS (white+gray+black 1.0–7.0 m, per-distance mean, quadratic),
written into results/calibration_methods.json by scripts/build_calibrations.py.
Usage: python scripts/stmap_eval.py [--workers 16]
       python scripts/stmap_eval.py --timing   # add 'stmap' to results/timing.json (same protocol as timing.py:
                                               # red test set, 30 images, read two images + mask → σ → curve, CPU single thread)
"""
import os, sys, re, glob, argparse
import numpy as np
import cv2
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cadfds_core as C

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
A = 22.0 / 1.4


def stmap_sigma(g1, g2):
    g1 = g1.astype(np.float64); g2 = g2.astype(np.float64)
    num = ((g1 - g2) ** 2).sum(); den = (cv2.Laplacian(g1, cv2.CV_64F, ksize=3) ** 2).sum()
    if den <= 0:
        return np.nan
    return float(abs(np.sqrt(16.0 * num / den) / (A ** 2 - 1)) ** 0.25)


def work(a):
    rf, mp, rot = a
    lf = rf.replace(os.sep + 'R' + os.sep, os.sep + 'L' + os.sep)
    g1 = cv2.imdecode(np.fromfile(rf, np.uint8), cv2.IMREAD_GRAYSCALE)
    g2 = cv2.imdecode(np.fromfile(lf, np.uint8), cv2.IMREAD_GRAYSCALE) if os.path.exists(lf) else None
    if g1 is None or g2 is None:
        return (np.nan, np.nan)
    m = C.load_mask(mp, g1.shape); ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = stmap_sigma(g1[y0:y1, x0:x1], g2[y0:y1, x0:x1])
    sr = np.nan
    if rot:
        r1, r2 = cv2.rotate(g1, cv2.ROTATE_180), cv2.rotate(g2, cv2.ROTATE_180)
        sr = stmap_sigma(r1[y0:y1, x0:x1], r2[y0:y1, x0:x1])
    return (s, sr)


def items(kind, colors, shots=10):
    out = []
    for c in colors:
        root = os.path.join(DATA, 'training', f'firetruck_{c}', DS) if kind == 'cal' else os.path.join(DATA, f'firetruck_{c}', DS)
        mdir = os.path.join(DATA, 'training', 'Training_dataset_mask') if kind == 'cal' else os.path.join(DATA, 'firetruck_black', DS + '_mask')
        for tdir in sorted(glob.glob(os.path.join(root, 'T*'))):
            t = int(re.search(r'T(\d+)$', tdir).group(1)); mp = os.path.join(mdir, f'T{t}.npy')
            if not os.path.exists(mp):
                continue
            for k, f in enumerate(sorted(glob.glob(os.path.join(tdir, 'R', '*.png')))[:shots], 1):
                out.append((c, t, k, float(C.DIST_RE.search(os.path.basename(f)).group(1)), f, mp))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--timing', action='store_true')
    a = ap.parse_args()
    if a.timing:
        import time, json
        it = items('test', ['red']); step = max(1, len(it) // 30); it = it[::step][:30]
        work((it[0][4], it[0][5], False)); ts = []
        for (_, _, _, _, f, mp) in it:
            t0 = time.perf_counter(); work((f, mp, False)); ts.append(time.perf_counter() - t0)
        tp = os.path.join(ROOT, 'results', 'timing.json'); tm = json.load(open(tp, encoding='utf-8'))
        tm['stmap'] = dict(mean_ms=round(1000 * np.mean(ts), 2), median_ms=round(1000 * np.median(ts), 2), fps=round(1 / np.mean(ts), 2),
                           note='two images (f/1.4 + f/22) read + mask box + sigma, CPU single thread (scripts/stmap_eval.py --timing)')
        json.dump(tm, open(tp, 'w', encoding='utf-8'), indent=2); print('stmap timing', tm['stmap']); sys.exit()
    for kind, colors in (('cal', ['white', 'gray', 'black']), ('test', ['red', 'green', 'blue', 'black'])):
        it = items(kind, colors)
        with Pool(a.workers) as pool:
            res = pool.map(work, [(f, mp, kind == 'test') for (_, _, _, _, f, mp) in it], chunksize=4)
        np.savez(os.path.join(ROOT, 'features', f'stmap_{kind}.npz'),
                 color=np.array([x[0] for x in it]), t=np.array([x[1] for x in it]), shot=np.array([x[2] for x in it]),
                 dist_cm=np.array([x[3] * 100 for x in it]), path=np.array([x[4] for x in it]),
                 sigma=np.array([r[0] for r in res]), sigma_rot=np.array([r[1] for r in res]))
        print(kind, len(it), 'pairs,', int(np.isnan([r[0] for r in res]).sum()), 'failed → features/stmap_%s.npz' % kind, flush=True)
