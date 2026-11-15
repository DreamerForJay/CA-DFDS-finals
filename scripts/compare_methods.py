# -*- coding: utf-8 -*-
"""Overall method comparison (CA-DFDS variants vs grey-scale DFD vs UniDepth V2 vs ML-Depth-Pro) → results/method_comparison.{csv,json,md}

Inputs:
  features/test_<color>.npz             CA-DFDS / grey-scale sharpness (task A)
  results/calibration_methods.json      calibration curves for each variant (scripts/build_calibrations.py)
  results/deep_baselines.csv            per-image predictions from UniDepth / Depth Pro (baselines/eval_deep.py)
  results/timing.json                   CPU timing (task D)
All methods use the same 480 images (red/green/blue/black × T01–T12 × 10 images, 1.2–6.7 m) and the same hand-drawn
mask; per-image protocol.
"""
import os, sys, csv, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cadfds_core as C

R = lambda *p: os.path.join(ROOT, *p)
CAL = json.load(open(R('results', 'calibration_methods.json'), encoding='utf-8'))
deep = list(csv.DictReader(open(R('results', 'deep_baselines.csv'), encoding='utf-8-sig')))
key = lambda p: os.path.normcase(os.path.normpath(p))
deep_by = {key(r['path']): r for r in deep}
colors = sorted({r['color'] for r in deep}, key=['red', 'green', 'blue', 'black'].index)
NAMES = {'opt': 'CA-DFDS (paper weights)', 'reopt5': 'CA-DFDS re-optimised eq.5', 'jonly': 'CA-DFDS J only',
         'gray': 'Grey-scale DFD', 'unidepth': 'UniDepth V2', 'depthpro': 'ML-Depth-Pro'}
rows = []   # per-image
for c in colors:
    z = np.load(R('features', f'test_{c}.npz'), allow_pickle=True)
    for i, p in enumerate(z['path']):
        d = deep_by.get(key(p))
        if d is None:
            continue
        r = dict(color=c, t=int(z['t'][i]), gt_m=float(z['dist_cm'][i]) / 100)
        for k in ('opt', 'reopt5', 'jonly', 'gray'):
            x = z['gray'][i] if k == 'gray' else C.sigma_eq5(z['cov'][i], CAL[k]['weights'])
            r[k] = float(np.polyval(CAL[k]['coef_cm'], x)) / 100
        r['unidepth'] = float(d['unidepth_m']); r['depthpro'] = float(d['depthpro_m'])
        r['sam_iou'] = float(d.get('sam_iou') or 'nan')
        rows.append(r)
M = list(NAMES)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
res = {'n_images': len(rows), 'colors': colors, 'per_color_rmse_m': {}, 'overall_rmse_m': {}, 'overall_mae_m': {},
       'overall_bias_m': {}, 'per_distance_rmse_m': {}, 'time_ms': {}}
for k in M:
    e = np.array([r[k] - r['gt_m'] for r in rows])
    res['overall_rmse_m'][k] = round(rm(e), 4); res['overall_mae_m'][k] = round(float(np.abs(e).mean()), 4)
    res['overall_bias_m'][k] = round(float(e.mean()), 4)
    res['per_color_rmse_m'][k] = {c: round(rm([r[k] - r['gt_m'] for r in rows if r['color'] == c]), 4) for c in colors}
    res['per_distance_rmse_m'][k] = {f'{g:.1f}': round(rm([r[k] - r['gt_m'] for r in rows if abs(r['gt_m'] - g) < 1e-6]), 4)
                                     for g in sorted({r['gt_m'] for r in rows})}
tm = json.load(open(R('results', 'timing.json'), encoding='utf-8'))
dt = {k: float(np.median([float(r[f'{k}_ms']) for r in deep])) for k in ('unidepth', 'depthpro')}
res['time_ms'] = {'opt': tm['ciecam02']['median_ms'], 'reopt5': tm['ciecam02']['median_ms'], 'jonly': tm['ciecam02']['median_ms'],
                  'gray': tm['gray']['median_ms'], 'unidepth': round(dt['unidepth'], 1), 'depthpro': round(dt['depthpro'], 1)}
res['time_note'] = 'CA-DFDS/grey-scale: CPU single thread, full image (results/timing.json); deep models: RTX 5080 GPU, inference only (median)'
if any(np.isfinite(r['sam_iou']) for r in rows):
    v = np.array([r['sam_iou'] for r in rows]); res['sam_box_iou_vs_hand_mask'] = {'mean': round(float(np.nanmean(v)), 4), 'min': round(float(np.nanmin(v)), 4)}
json.dump(res, open(R('results', 'method_comparison.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
with open(R('results', 'method_comparison_per_image.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# Markdown table
L = ['| Method | ' + ' | '.join(colors) + ' | Overall RMSE | MAE | Mean bias | Time per image |', '|---' * (len(colors) + 5) + '|']
for k in M:
    L.append(f'| {NAMES[k]} | ' + ' | '.join(f"{res['per_color_rmse_m'][k][c]:.3f}" for c in colors) +
             f" | **{res['overall_rmse_m'][k]:.3f}** | {res['overall_mae_m'][k]:.3f} | {res['overall_bias_m'][k]:+.3f} | {res['time_ms'][k]:.0f} ms |")
open(R('results', 'method_comparison.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L)); print(json.dumps(res['per_distance_rmse_m'], ensure_ascii=False)); print(res.get('sam_box_iou_vs_hand_mask'))
