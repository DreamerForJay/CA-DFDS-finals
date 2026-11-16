# -*- coding: utf-8 -*-
"""M3 / issue #1: segmented timing of the full CA-DFDS pipeline on the lab RTX 5080 host.

Segments: read image | mask (hand mask load, or automatic YOLO11m -> SAM ViT-B) | crop to mask box |
          3x3 Gaussian | sRGB -> CIECAM02 | Laplacian + masked covariance | sigma + calibration curve.
Versions: CPU (colour-science, current code) full frame / cropped; GPU (cadfds_gpu.py, torch) float64 / float32, full / cropped.
Images: test set red, 30 images spread evenly over T01-T12 (1.2-6.7 m), 3 warm-up images not counted.

The CPU part needs colour-science (.venv-cadfds); the GPU and mask parts need torch (../.venv). Run:
  .venv-cadfds/Scripts/python.exe scripts/m3_timing_full_pipeline.py --part cpu
  ../.venv/Scripts/python.exe     scripts/m3_timing_full_pipeline.py --part gpu
  ../.venv/Scripts/python.exe     scripts/m3_timing_full_pipeline.py --part mask
  .venv-cadfds/Scripts/python.exe scripts/m3_timing_full_pipeline.py --merge     -> results/timing_full_pipeline.json
"""
import os, sys, json, time, argparse, platform
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cadfds_core as C

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = os.path.join(DATA, 'firetruck_red', 'Test_dataset_Focus1500cm_f1.4_f22.0')
MASK = os.path.join(DATA, 'firetruck_black', 'Test_dataset_Focus1500cm_f1.4_f22.0_mask')
CO = np.array(json.load(open(os.path.join(ROOT, 'results', 'calibration_opt.json')))['coef_cm'])
W = np.array(C.W_OPT)
SCR = os.path.join(ROOT, 'logs', 'scratch')
N, WARM = 30, 3


def items():
    it = [x for x in C.discover(DS) if C.find_mask([MASK], x[1])]
    step = max(1, len(it) // N)
    return it[:WARM] + it[::step][:N]


def stats(ts):
    ts = np.array(ts) * 1000
    return dict(mean_ms=round(float(ts.mean()), 2), median_ms=round(float(np.median(ts)), 2), min_ms=round(float(ts.min()), 2),
                max_ms=round(float(ts.max()), 2))


def summarize(recs):
    out = {k: stats([r[k] for r in recs[WARM:]]) for k in recs[0] if k not in ('path', 'dist_cm', 'pred_cm', 'pixels')}
    tot = np.array([r['total'] for r in recs[WARM:]])
    out['fps'] = round(float(1 / tot.mean()), 2)
    out['rmse_m'] = round(float(np.sqrt(np.mean([(r['pred_cm'] - r['dist_cm']) ** 2 for r in recs[WARM:]])) / 100), 4)
    out['mean_pixels'] = int(np.mean([r['pixels'] for r in recs[WARM:]]))
    return out


def run_cpu():
    import cv2, colour
    from cadfds_core import ciecam_channels
    res = {}
    for crop in (False, True):
        recs = []
        for p, t, d in items():
            r = {'path': p, 'dist_cm': d}; T = time.perf_counter
            t0 = T(); img = C.imread(p); r['read'] = T() - t0
            t0 = T(); m = C.load_mask(C.find_mask([MASK], t), img.shape); r['mask_load'] = T() - t0
            t0 = T()
            if crop:
                img, m = C.crop_to_mask(img, m)
            r['crop'] = T() - t0; r['pixels'] = img.shape[0] * img.shape[1]
            t0 = T(); blur = cv2.GaussianBlur(img, (3, 3), 0); r['blur'] = T() - t0
            t0 = T()
            rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            XYZ = colour.sRGB_to_XYZ(rgb)
            XYZ_w = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']) * 100
            spec = colour.XYZ_to_CIECAM02(XYZ * 100, XYZ_w, 2000 / np.pi * 0.2, 20.0, surround=colour.VIEWING_CONDITIONS_CIECAM02['Average'])
            J = np.nan_to_num(spec.J); Cc = np.nan_to_num(spec.C); h = np.deg2rad(np.nan_to_num(spec.h))
            ch = (J, Cc * np.cos(h), Cc * np.sin(h)); r['ciecam02'] = T() - t0
            t0 = T()
            L = [cv2.Laplacian(x, cv2.CV_64F, ksize=3) for x in ch]
            cov = np.cov(np.stack([l[m] for l in L]), bias=True); r['laplacian_cov'] = T() - t0
            t0 = T(); s = C.sigma_eq5(cov, W); r['pred_cm'] = float(np.polyval(CO, s)); r['curve'] = T() - t0
            r['total'] = sum(r[k] for k in ('read', 'mask_load', 'crop', 'blur', 'ciecam02', 'laplacian_cov', 'curve'))
            recs.append(r)
        res['cpu_crop' if crop else 'cpu_full'] = summarize(recs)
        print('cpu crop' if crop else 'cpu full', res['cpu_crop' if crop else 'cpu_full']['total'], flush=True)
    return res


def run_gpu():
    import cv2, torch
    import cadfds_gpu as G
    res = {}
    sync = torch.cuda.synchronize
    for dtn, dt in (('fp64', torch.float64), ('fp32', torch.float32)):
        g = G.CIECAM02GPU(dtype=dt)
        for crop in (False, True):
            recs = []
            for p, t, d in items():
                r = {'path': p, 'dist_cm': d}; T = time.perf_counter
                t0 = T(); img = C.imread(p); r['read'] = T() - t0
                t0 = T(); m = C.load_mask(C.find_mask([MASK], t), img.shape); r['mask_load'] = T() - t0
                t0 = T()
                if crop:
                    img, m = C.crop_to_mask(img, m)
                r['crop'] = T() - t0; r['pixels'] = img.shape[0] * img.shape[1]
                t0 = T(); blur = cv2.GaussianBlur(img, (3, 3), 0); r['blur'] = T() - t0
                sync(); t0 = T()
                rgb = torch.from_numpy(np.ascontiguousarray(blur[..., ::-1])).to('cuda'); mt = torch.from_numpy(m).to('cuda')
                sync(); r['upload'] = T() - t0
                t0 = T(); ch = g.channels(rgb); sync(); r['ciecam02'] = T() - t0
                t0 = T(); cov = g.laplacian_cov(ch, mt).double().cpu().numpy(); r['laplacian_cov'] = T() - t0
                t0 = T(); s = C.sigma_eq5(cov, W); r['pred_cm'] = float(np.polyval(CO, s)); r['curve'] = T() - t0
                r['total'] = sum(r[k] for k in ('read', 'mask_load', 'crop', 'blur', 'upload', 'ciecam02', 'laplacian_cov', 'curve'))
                recs.append(r)
            k = f'gpu_{dtn}_{"crop" if crop else "full"}'; res[k] = summarize(recs); print(k, res[k]['total'], flush=True)
    res['gpu_name'] = torch.cuda.get_device_name(0); res['torch'] = torch.__version__; res['cuda'] = torch.version.cuda
    return res


def run_mask():
    import torch
    sys.path.insert(0, os.path.join(ROOT, 'baselines'))
    import deep_worker as Wk
    Wk._lazy_imports()
    Wk.handle({'cmd': 'warmup', 'models': ['sam', 'yolo']})
    out = os.path.join(SCR, 'm3_auto_mask.npy'); recs = []
    for p, t, d in items():
        t0 = time.perf_counter(); o = Wk.auto(p, out); wall = time.perf_counter() - t0
        h = C.load_mask(C.find_mask([MASK], t)); a = np.load(out) > 0
        recs.append(dict(source=o['source'], det=o['det_ms'] / 1000, sam=o['sam_ms'] / 1000, wall=wall,
                         iou=float((a & h).sum() / (a | h).sum())))
    R = recs[WARM:]
    res = {'auto_mask_yolo_sam': {
        'yolo_detect': stats([r['det'] for r in R]), 'sam': stats([r['sam'] for r in R]), 'wall_incl_io': stats([r['wall'] for r in R]),
        'fallback_count': sum(r['source'] == 'fallback' for r in R), 'iou_mean': round(float(np.mean([r['iou'] for r in R])), 4),
        'note': 'sam = SAM image encoding + box decoding; fallback (YOLO miss, < ~3 m) runs one encoding and two box decodings'}}
    print(res, flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', choices=['cpu', 'gpu', 'mask']); ap.add_argument('--merge', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'timing_full_pipeline.json')); a = ap.parse_args()
    os.makedirs(SCR, exist_ok=True)
    if a.part:
        r = {'cpu': run_cpu, 'gpu': run_gpu, 'mask': run_mask}[a.part]()
        r['host'] = platform.node(); r['python'] = platform.python_version()
        json.dump(r, open(os.path.join(SCR, f'm3_{a.part}.json'), 'w'), indent=2)
    if a.merge:
        out = {'images': f'firetruck_red test set, {N} images over T01-T12 (1.2-6.7 m), {WARM} warm-up excluded',
               'cpu': platform.processor(), 'units': 'ms per image'}
        for part in ('cpu', 'gpu', 'mask'):
            out[part] = json.load(open(os.path.join(SCR, f'm3_{part}.json')))
        json.dump(out, open(a.out, 'w'), indent=2); print('saved', a.out)
