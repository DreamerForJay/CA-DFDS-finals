# -*- coding: utf-8 -*-
"""Recompute CA-DFDS / grayscale sharpness with the fully automatic masks (YOLO→SAM, produced by baselines/eval_deep.py --auto)
→ features/test_auto_<color>.npz (same format as test_<color>.npz)

Purpose: in real road scenes there are no hand-drawn masks. This quantifies how much the distance error changes once masks are generated automatically (NEXT_EXPERIMENTS M4).
Usage: python scripts/eval_auto_mask.py [--workers 16]
"""
import os, sys, csv, argparse
import numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cadfds_core as C


def work(a):
    path, mpath = a
    img = C.imread(path)
    if img is None or not mpath or not os.path.exists(mpath):
        return None
    m = C.load_mask(mpath, img.shape)
    if m.sum() < 100:
        return None
    ci, cm = C.crop_to_mask(img, m)
    return C.laplacian_cov(ci, cm), C.gray_sigma(ci, cm)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=16); a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results', 'deep_baselines.csv'), encoding='utf-8-sig')))
    for c in sorted({r['color'] for r in rows}):
        rs = [r for r in rows if r['color'] == c]
        with Pool(a.workers) as pool:
            res = pool.map(work, [(r['path'], r.get('auto_mask')) for r in rs], chunksize=2)
        np.savez(os.path.join(ROOT, 'features', f'test_auto_{c}.npz'),
                 cov=np.array([x[0] if x else np.full((3, 3), np.nan) for x in res]),
                 gray=np.array([x[1] if x else np.nan for x in res]),
                 dist_cm=np.array([float(r['dist_m']) * 100 for r in rs]), t=np.array([int(r['t']) for r in rs]),
                 path=np.array([r['path'] for r in rs]))
        print(c, len(rs), 'images,', sum(x is None for x in res), 'failed → features/test_auto_%s.npz' % c, flush=True)
