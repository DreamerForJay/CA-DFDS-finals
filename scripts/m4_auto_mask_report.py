# -*- coding: utf-8 -*-
"""M4 / issue #5 (part 2, .venv-cadfds): effect of automatic masks on distance error, incl. a FULLY automatic pipeline
(calibration curve also fitted on automatic masks).

Inputs : results/deep_baselines.csv (test set auto masks, from baselines/eval_deep.py --auto),
         logs/scratch/auto_masks_calib.csv (calibration auto masks, from scripts/m4_auto_mask_calib.py)
Outputs: features/cal_auto_{white,gray,black}.npz
         results/auto_mask.csv            per test image: mask source / IoU / time, prediction under 3 protocols
         results/auto_mask_summary.json   RMSE table + IoU / timing summary
         results/auto_mask_examples/*.jpg hand-mask (green) vs automatic mask (red) contours
Protocols: hand    = calibration hand masks, test hand masks (paper)
           auto    = calibration hand masks, test automatic masks (RESULTS J)
           fullauto= calibration automatic masks, test automatic masks (no hand mask anywhere)
"""
import os, sys, csv, json, argparse
import numpy as np
import cv2
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import cadfds_core as C
from evaluate import load, per_dist
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_auto_mask import work  # same per-image routine (crop to mask box, CIECAM02 cov + gray sigma)

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
HAND_TEST = os.path.join(DATA, 'firetruck_black', 'Test_dataset_Focus1500cm_f1.4_f22.0_mask')
SETTINGS = {'gray': 'gray', 'opt(0.8136,0.1782,0.0081)': C.W_OPT, 'Jonly(1,0,0)': (1.0, 0.0, 0.0)}


def load2(f):
    D = load(f); z = np.load(f, allow_pickle=True); D['path'] = z['path']; D['t'] = z['t']
    return D


def feat(D, s):
    return D['gray'] if isinstance(s, str) else C.sigma_eq5(D['cov'], s)


def curve(cal, s):
    X, Y = [], []
    for D in cal:
        x, d = per_dist(feat(D, s), D['dist']); ok = np.isfinite(x); X += list(x[ok]); Y += list(d[ok])
    return C.fit_quadratic(np.array(X), np.array(Y))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=16); a = ap.parse_args()
    crow = list(csv.DictReader(open('logs/scratch/auto_masks_calib.csv', encoding='utf-8-sig')))
    for c in ('white', 'gray', 'black'):
        f = f'features/cal_auto_{c}.npz'
        if os.path.exists(f):
            continue
        rs = [r for r in crow if r['color'] == c]
        with Pool(a.workers) as pool:
            res = pool.map(work, [(r['path'], r['auto_mask']) for r in rs], chunksize=2)
        np.savez(f, cov=np.array([x[0] if x else np.full((3, 3), np.nan) for x in res]),
                 gray=np.array([x[1] if x else np.nan for x in res]), dist_cm=np.array([float(r['dist_m']) * 100 for r in rs]),
                 t=np.array([int(r['t']) for r in rs]), path=np.array([r['path'] for r in rs]))
        print('saved', f, flush=True)
    cal_hand = [load(f'features/cal_{c}.npz') for c in ('white', 'gray', 'black')]
    cal_auto = [load(f'features/cal_auto_{c}.npz') for c in ('white', 'gray', 'black')]
    colors = ('red', 'green', 'blue', 'black')
    th = {c: load2(f'features/test_{c}.npz') for c in colors}; ta = {c: load2(f'features/test_auto_{c}.npz') for c in colors}
    trow = list(csv.DictReader(open('results/deep_baselines.csv', encoding='utf-8-sig')))

    summary = {'rmse_m': {}, 'calib_r2': {}}
    pred = {}
    for sn, s in SETTINGS.items():
        co_h, r2h = curve(cal_hand, s); co_a, r2a = curve(cal_auto, s)
        summary['calib_r2'][sn] = {'hand': round(r2h, 4), 'auto': round(r2a, 4)}
        out = {}
        for proto, co, T in (('hand', co_h, th), ('auto', co_h, ta), ('fullauto', co_a, ta)):
            err = []
            for c in colors:
                D = T[c]; p = np.polyval(co, feat(D, s)) / 100
                # map (t, path) -> prediction; hand features include T13 rows without mask (nan) -> skip
                for path, t, d, v in zip(D['path'], D['t'], D['dist'], p):
                    if int(t) <= 12 and np.isfinite(v):
                        pred[(sn, proto, os.path.normcase(os.path.normpath(str(path))))] = float(v); err.append(v - d / 100)
            out[proto] = round(float(np.sqrt(np.mean(np.square(err)))), 4)
            out[proto + '_n'] = len(err)
        summary['rmse_m'][sn] = out
        print(sn, out, summary['calib_r2'][sn], flush=True)

    rows = []
    for r in trow:
        k = os.path.normcase(os.path.normpath(r['path']))
        row = dict(set='test', color=r['color'], t=r['t'], shot=r['shot'], dist_m=r['dist_m'], auto_source=r['auto_source'],
                   auto_iou=r['auto_iou'], yolo_ms=r['yolo_ms'], auto_sam_ms=r['auto_sam_ms'])
        for sn in SETTINGS:
            tag = 'gray' if sn == 'gray' else ('cadfds' if sn.startswith('opt') else 'jonly')
            for proto in ('hand', 'auto', 'fullauto'):
                v = pred.get((sn, proto, k)); row[f'{tag}_{proto}_m'] = round(v, 4) if v is not None else ''
        rows.append(row)
    for r in crow:
        rows.append(dict(set='calib', color=r['color'], t=r['t'], shot=r['shot'], dist_m=r['dist_m'], auto_source=r['auto_source'],
                         auto_iou=r['auto_iou'], yolo_ms=r['yolo_ms'], auto_sam_ms=r['auto_sam_ms']))
    keys = list(rows[0])
    with open('results/auto_mask.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)

    for name, rs in (('test', trow), ('calib', crow)):
        iou = np.array([float(r['auto_iou']) for r in rs]); src = np.array([r['auto_source'] for r in rs])
        summary[f'iou_{name}'] = dict(n=len(rs), mean=round(float(iou.mean()), 4), min=round(float(iou.min()), 4),
                                      yolo_mean=round(float(iou[src == 'yolo'].mean()), 4), fallback_mean=round(float(iou[src == 'fallback'].mean()), 4),
                                      fallback_count=int((src == 'fallback').sum()))
        summary[f'time_ms_{name}'] = dict(yolo_mean=round(float(np.mean([float(r['yolo_ms']) for r in rs])), 1),
                                          sam_mean=round(float(np.mean([float(r['auto_sam_ms']) for r in rs])), 1))
    # per-distance RMSE, CA-DFDS paper weights
    pdist = {}
    for proto in ('hand', 'auto', 'fullauto'):
        for d in sorted({float(r['dist_m']) for r in rows if r['set'] == 'test'}):
            e = [float(r[f'cadfds_{proto}_m']) - d for r in rows if r['set'] == 'test' and float(r['dist_m']) == d and r[f'cadfds_{proto}_m'] != '']
            pdist.setdefault(str(d), {})[proto] = round(float(np.sqrt(np.mean(np.square(e)))), 3)
    summary['cadfds_rmse_by_distance_m'] = pdist
    json.dump(summary, open('results/auto_mask_summary.json', 'w'), indent=2)
    print(json.dumps({k: summary[k] for k in summary if k.startswith(('iou', 'time'))}, indent=1))

    # example overlays: red car, shot 1, five distances + the worst test IoU
    ex = 'results/auto_mask_examples'; os.makedirs(ex, exist_ok=True)
    pick = [r for r in trow if r['color'] == 'red' and r['shot'] == '1' and r['t'] in ('1', '3', '5', '8', '12')]
    pick.append(min(trow, key=lambda r: float(r['auto_iou'])))
    for r in pick:
        img = C.imread(r['path']); h = C.load_mask(C.find_mask([HAND_TEST], int(r['t'])), img.shape)
        am = C.load_mask(r['auto_mask'], img.shape); vis = img.copy()
        for m, col in ((h, (0, 200, 0)), (am, (0, 0, 255))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cs, -1, col, 6)
        cv2.putText(vis, f"{r['color']} {r['dist_m']} m  {r['auto_source']}  IoU {float(r['auto_iou']):.3f}  (green=hand, red=auto)",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
        vis = cv2.resize(vis, (1024, 768), interpolation=cv2.INTER_AREA)
        fn = f"{ex}/{r['color']}_T{int(r['t']):02d}_shot{int(r['shot']):02d}_{r['auto_source']}.jpg"
        cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(fn)
    print('saved results/auto_mask.csv, results/auto_mask_summary.json,', len(pick), 'examples')
