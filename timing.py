# -*- coding: utf-8 -*-
"""量測每張影像處理時間：讀圖 → 讀遮罩 → 計算清晰度 → 校正曲線預測（不含遮罩取得）。
同時量灰階版本與 CIECAM02 最終版本，請在原量測 74.41 ms 之同一台電腦（RTX 5080 主機）執行。
用法：python timing.py --dataset <資料集> --mask_dir <遮罩> --n 30 --out results/timing.json
"""
import argparse, time, json, platform, os, numpy as np
import cadfds_core as C
ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--mask_dir', nargs='+', required=True)
ap.add_argument('--n', type=int, default=30); ap.add_argument('--out', default='results/timing.json'); a = ap.parse_args()
items = C.discover(a.dataset); step = max(1, len(items) // a.n); items = items[::step][:a.n]
CO = (-6.562, 128.83, 68.39)
def run(path, t, mode):
    img = C.imread(path); m = C.load_mask(C.find_mask(a.mask_dir, t), img.shape)
    s = C.gray_sigma(img, m) if mode == 'gray' else C.sharpness(img, m, C.W_OPT)
    return np.polyval(CO, s)
res = {'cpu': platform.processor(), 'machine': platform.node(), 'n': len(items)}
for mode in ('gray', 'ciecam02'):
    run(items[0][0], items[0][1], mode); ts = []
    for p, t, _ in items:
        t0 = time.perf_counter(); run(p, t, mode); ts.append(time.perf_counter() - t0)
    res[mode] = dict(mean_ms=round(1000 * np.mean(ts), 2), median_ms=round(1000 * np.median(ts), 2), fps=round(1 / np.mean(ts), 2))
    print(mode, res[mode])
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=2); print(res)
