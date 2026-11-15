# -*- coding: utf-8 -*-
"""Overall method comparison → results/method_comparison.{csv,json,md}

Methods (all use the same 480 test images: red/green/blue/black × T01–T12 × 10 shots, 1.2–6.7 m, each image estimated on its own):
  CA-DFDS paper weights / re-optimized eq. 5 / J only, grayscale DFD  — quadratic calibration curve fitted on the calibration set
  UniDepth V2, ML-Depth-Pro — raw (official weights, uncalibrated) and linearly calibrated (same calibration set, d = a·pred + b)
Two mask conditions:
  hand — hand-drawn masks (the setting used for all experiments so far)
  auto — fully automatic masks (YOLO11→SAM, close-range fallback; baselines/eval_deep.py --auto + scripts/eval_auto_mask.py)
Inputs: features/test_<c>.npz, features/test_auto_<c>.npz, results/calibration_methods.json, results/deep_baselines.csv,
        results/timing.json
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
NAMES = {'opt': 'CA-DFDS (paper weights)', 'reopt5': 'CA-DFDS re-optimized eq. 5', 'jonly': 'CA-DFDS J only',
         'gray': 'Grayscale DFD', 'unidepth': 'UniDepth V2 (raw)', 'unidepth_lin': 'UniDepth V2 (linear calib.)',
         'depthpro': 'ML-Depth-Pro (raw)', 'depthpro_lin': 'ML-Depth-Pro (linear calib.)'}
M = [k for k in NAMES if k in CAL or k in ('unidepth', 'depthpro')]
fnum = lambda v: float(v) if v not in (None, '') else np.nan


def predict(z, i, d, mask):
    out = {}
    if z is not None:
        for k in ('opt', 'reopt5', 'jonly', 'gray'):
            x = z['gray'][i] if k == 'gray' else C.sigma_eq5(z['cov'][i], CAL[k]['weights'])
            out[k] = float(np.polyval(CAL[k]['coef_cm'], x)) / 100 if np.isfinite(x) else np.nan
    suf = '' if mask == 'hand' else '_auto'
    for k in ('unidepth', 'depthpro'):
        raw = fnum(d.get(f'{k}{suf}_m'))
        out[k] = raw
        if k + '_lin' in CAL:
            out[k + '_lin'] = float(np.polyval(CAL[k + '_lin']['coef_cm'], raw)) / 100 if np.isfinite(raw) else np.nan
    return out


rows = []
for mask in ('hand', 'auto'):
    for c in colors:
        f = R('features', f'test_{c}.npz' if mask == 'hand' else f'test_auto_{c}.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        for i, p in enumerate(z['path']):
            d = deep_by.get(key(p))
            if d is None:
                continue
            r = dict(mask=mask, color=c, t=int(z['t'][i]), gt_m=float(z['dist_cm'][i]) / 100,
                     auto_source=d.get('auto_source', ''), auto_iou=fnum(d.get('auto_iou')), sam_iou=fnum(d.get('sam_iou')))
            r.update(predict(z, i, d, mask))
            rows.append(r)

rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))) if len(e) else np.nan
res = {'n_images': sum(r['mask'] == 'hand' for r in rows), 'colors': colors, 'methods': NAMES}
for mask in ('hand', 'auto'):
    rs = [r for r in rows if r['mask'] == mask]
    if not rs:
        continue
    blk = {'overall_rmse_m': {}, 'overall_mae_m': {}, 'overall_bias_m': {}, 'per_color_rmse_m': {}, 'per_distance_rmse_m': {}, 'n_failed': {}}
    for k in M:
        e = np.array([r[k] - r['gt_m'] for r in rs])
        blk['n_failed'][k] = int(np.isnan(e).sum())
        blk['overall_rmse_m'][k] = round(rm(e), 4); blk['overall_mae_m'][k] = round(float(np.nanmean(np.abs(e))), 4)
        blk['overall_bias_m'][k] = round(float(np.nanmean(e)), 4)
        blk['per_color_rmse_m'][k] = {c: round(rm(np.array([r[k] - r['gt_m'] for r in rs if r['color'] == c])), 4) for c in colors}
        blk['per_distance_rmse_m'][k] = {f'{g:.1f}': round(rm(np.array([r[k] - r['gt_m'] for r in rs if abs(r['gt_m'] - g) < 1e-6])), 4)
                                         for g in sorted({r['gt_m'] for r in rs})}
    res[mask] = blk
res['overall_rmse_m'] = res['hand']['overall_rmse_m']   # GUI compatibility

# automatic-mask quality
if 'auto' in res:
    a = [r for r in rows if r['mask'] == 'hand']
    src = {}
    for r in a:
        s = r['auto_source'] or 'none'; src.setdefault(s, []).append(r['auto_iou'])
    res['auto_mask_quality'] = {s: {'n': len(v), 'iou_mean': round(float(np.nanmean(v)), 4), 'iou_min': round(float(np.nanmin(v)), 4)}
                                for s, v in src.items()}
    v = np.array([r['auto_iou'] for r in a]); res['auto_mask_quality']['all'] = {'n': len(v), 'iou_mean': round(float(np.nanmean(v)), 4),
                                                                                 'iou_min': round(float(np.nanmin(v)), 4)}
    det = {}
    for r in a:
        det.setdefault(f"{r['gt_m']:.1f}", []).append(r['auto_source'] == 'yolo')
    res['yolo_detect_rate_by_distance'] = {k: round(float(np.mean(v)), 3) for k, v in det.items()}
v = np.array([r['sam_iou'] for r in rows if r['mask'] == 'hand'])
if np.isfinite(v).any():
    res['sam_box_iou_vs_hand_mask'] = {'mean': round(float(np.nanmean(v)), 4), 'min': round(float(np.nanmin(v)), 4)}

tm = json.load(open(R('results', 'timing.json'), encoding='utf-8'))
med = lambda col: round(float(np.median([fnum(r.get(col)) for r in deep if r.get(col) not in (None, '')])), 1)
res['time_ms'] = {'opt': tm['ciecam02']['median_ms'], 'reopt5': tm['ciecam02']['median_ms'], 'jonly': tm['ciecam02']['median_ms'],
                  'gray': tm['gray']['median_ms'], 'unidepth': med('unidepth_ms'), 'unidepth_lin': med('unidepth_ms'),
                  'depthpro': med('depthpro_ms'), 'depthpro_lin': med('depthpro_ms')}
if 'yolo_ms' in deep[0]:
    res['time_ms_mask'] = {'yolo': med('yolo_ms'), 'sam': med('auto_sam_ms')}
res['time_note'] = 'CA-DFDS/grayscale: CPU single thread, full image (results/timing.json); deep models: RTX 5080 GPU, inference only (median)'
json.dump(res, open(R('results', 'method_comparison.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(R('results', 'method_comparison_per_image.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

L = []
for mask, title in (('hand', 'Hand-drawn masks'), ('auto', 'Fully automatic masks (YOLO11→SAM)')):
    if mask not in res:
        continue
    b = res[mask]
    L += [f'### {title}', '', '| Method | ' + ' | '.join(colors) + ' | Overall RMSE | MAE | Mean bias | Time per image |', '|---' * (len(colors) + 5) + '|']
    for k in M:
        L.append(f'| {NAMES[k]} | ' + ' | '.join(f"{b['per_color_rmse_m'][k][c]:.3f}" for c in colors) +
                 f" | **{b['overall_rmse_m'][k]:.3f}** | {b['overall_mae_m'][k]:.3f} | {b['overall_bias_m'][k]:+.3f} | {res['time_ms'][k]:.0f} ms |")
    L.append('')
open(R('results', 'method_comparison.md'), 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
for k in ('auto_mask_quality', 'yolo_detect_rate_by_distance', 'sam_box_iou_vs_hand_mask', 'time_ms_mask'):
    print(k, res.get(k))
