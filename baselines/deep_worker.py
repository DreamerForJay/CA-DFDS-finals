# -*- coding: utf-8 -*-
"""Deep-learning comparison methods and SAM segmentation (run in a separate process with the torch environment).

Why a separate process: torch/SAM/UniDepth/Depth Pro are installed in D:\\Carla-0.10.0-Win64-Shipping\\.venv
(torch 2.11 + CUDA 12.8, supports the RTX 5080). The CA-DFDS core needs colour-science 0.4.6 in .venv-cadfds.
Keeping the two environments separate avoids package conflicts. gui_new_v3.py talks to this program over stdin/stdout JSON lines.

Protocol (one JSON object per line):
  {"cmd": "info"}                                          → hardware and environment info
  {"cmd": "sam", "image": p, "point": [x, y], "out": npy}  → SAM ViT-B single-point segmentation, mask saved to the out .npy
  {"cmd": "depth", "model": "unidepth"|"depthpro", "image": p, "mask": npy}
                                                           → median/mean metric depth inside the mask (m) and inference time
  {"cmd": "detect", "image": p}                            → YOLO11 vehicle boxes [{box, cls, conf}] (car/bus/truck/motorcycle)
  {"cmd": "auto", "image": p, "out": npy}                  → YOLO picks the largest vehicle box → SAM segments it by box (fully automatic mask, no manual work)
  {"cmd": "quit"}
image may also be a .npy (H×W×3 BGR array, used for video frames to skip PNG encoding).
Every reply carries "ok": true/false; on failure it carries "error".

Can also be run directly for testing: python deep_worker.py --selftest <image> <mask.npy>
"""
import os, sys, json, math, time, platform, traceback

BASE = os.environ.get('CADFDS_MODELS_DIR', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping')
SAM_CKPT = os.path.join(BASE, 'models', 'sam_vit_b_01ec64.pth')
DEPTHPRO_REPO = os.path.join(BASE, 'models_depthpro_repo')
UNIDEPTH_REPO = os.path.join(BASE, 'models_unidepth_repo')
UNIDEPTH_ID = 'lpiccinelli/unidepth-v2-vits14'
YOLO_W = os.path.join(BASE, 'models', os.environ.get('CADFDS_YOLO', 'yolo11m.pt'))
VEHICLE = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}   # COCO classes

# CARLA camera: 2048x1536, 40 mm lens, 36 mm sensor width (get_dataset2.py)
def f_px(width, lens_mm=40.0, sensor_mm=36.0):
    fov = 2 * math.atan(sensor_mm / (2 * lens_mm))
    return (width / 2.0) / math.tan(fov / 2.0)

_models = {}
np = cv2 = torch = None


def _lazy_imports():
    global np, cv2, torch
    if torch is None:
        import numpy as _np, cv2 as _cv2, torch as _torch
        np, cv2, torch = _np, _cv2, _torch


def device():
    _lazy_imports()
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def imread(p):
    _lazy_imports()
    if p.lower().endswith('.npy'):
        return np.load(p)
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def info():
    _lazy_imports()
    d = {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': device(),
         'cpu': platform.processor()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), cuda=torch.version.cuda,
                 sm=f'{p.major}.{p.minor}')
    d['models'] = {'sam': os.path.exists(SAM_CKPT),
                   'depthpro': os.path.exists(os.path.join(DEPTHPRO_REPO, 'checkpoints', 'depth_pro.pt')),
                   'unidepth': True, 'yolo': os.path.exists(YOLO_W)}
    return d


def get_sam():
    if 'sam' not in _models:
        from segment_anything import sam_model_registry, SamPredictor
        m = sam_model_registry['vit_b'](checkpoint=SAM_CKPT).to(device()).eval()
        _models['sam'] = SamPredictor(m)
    return _models['sam']


def get_depthpro():
    if 'depthpro' not in _models:
        sys.path.insert(0, os.path.join(DEPTHPRO_REPO, 'src'))
        import dataclasses, depth_pro
        cfg = dataclasses.replace(depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT,
                                  checkpoint_uri=os.path.join(DEPTHPRO_REPO, 'checkpoints', 'depth_pro.pt'))
        prec = torch.float16 if device() == 'cuda' else torch.float32
        m, tf = depth_pro.create_model_and_transforms(config=cfg, device=torch.device(device()), precision=prec)
        _models['depthpro'] = (m.eval(), tf)
    return _models['depthpro']


def get_unidepth():
    if 'unidepth' not in _models:
        if os.path.isdir(UNIDEPTH_REPO):
            sys.path.insert(0, UNIDEPTH_REPO)
        from unidepth.models import UniDepthV2
        m = UniDepthV2.from_pretrained(UNIDEPTH_ID)
        m.interpolation_mode = 'bilinear'
        _models['unidepth'] = m.to(device()).eval()
    return _models['unidepth']


def get_yolo():
    if 'yolo' not in _models:
        from ultralytics import YOLO
        _models['yolo'] = YOLO(YOLO_W)
    return _models['yolo']


def detect(image, conf=0.25, img=None):
    img = imread(image) if img is None else img
    _sync(); t0 = time.perf_counter()
    r = get_yolo().predict(img, conf=conf, classes=list(VEHICLE), verbose=False, device=0 if device() == 'cuda' else 'cpu')[0]
    _sync(); ms = 1000 * (time.perf_counter() - t0)
    dets = [{'box': [float(v) for v in b.xyxy[0].tolist()], 'cls': VEHICLE.get(int(b.cls), str(int(b.cls))), 'conf': float(b.conf)}
            for b in r.boxes]
    dets.sort(key=lambda d: -(d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]))   # largest (usually nearest) first
    return {'dets': dets, 'ms': ms}


FALLBACK_BOXES = [(0.0, 0.0, 1.0, 1.0), (0.03, 0.02, 0.97, 0.85)]   # relative coordinates: full frame, centre-lower region


def auto(image, out, prev_box=None):
    """Fully automatic mask. Three-level strategy:
    1. YOLO11 detects vehicles (car/bus/truck/motorcycle) → take the largest box (usually the nearest vehicle ahead) → SAM by box.
    2. YOLO finds nothing but the previous frame had a box (video) → previous box expanded 5% → SAM (tracking).
    3. Neither → near-range vehicle-ahead assumption: under 2.7 m the car fills the frame (only the rear panel, no wheels), which YOLO
       cannot recognise as a vehicle → run SAM with a full-frame box and a centre-lower box and keep the higher-scoring mask.
       Tested 2026-11-16: IoU 0.99 at 1.2 m, 0.91–0.96 at 1.7–3.2 m.
    """
    img = imread(image); H, W = img.shape[:2]; d = detect(image, img=img)
    if d['dets']:
        b = d['dets'][0]['box']; src = 'yolo'
        r = sam(image, [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], out, b, img=img)
    elif prev_box is not None:
        x0, y0, x1, y1 = prev_box; dx, dy = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
        b = [max(x0 - dx, 0), max(y0 - dy, 0), min(x1 + dx, W), min(y1 + dy, H)]; src = 'track'
        r = sam(image, [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], out, b, img=img)
    else:
        src = 'fallback'; best = None
        for i, fb in enumerate(FALLBACK_BOXES):
            bb = [fb[0] * W, fb[1] * H, fb[2] * W, fb[3] * H]
            rr = sam(image, [W / 2, H * 0.4], out + f'.{i}.npy', bb, img=img, reuse=i > 0)
            if best is None or rr['score'] > best[0]['score']:
                best = (rr, bb)
        r, b = best; os.replace(r['mask'], out); r['mask'] = out
        for i in range(len(FALLBACK_BOXES)):
            p = out + f'.{i}.npy'
            if os.path.exists(p):
                os.remove(p)
    m = np.load(out) > 0
    if m.any():   # report the mask's actual bounding box (used for tracking in the next video frame)
        ys, xs = np.nonzero(m); mbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    else:
        mbox = b
    return {'found': True, 'source': src, 'dets': d['dets'], 'box': [float(v) for v in b], 'mask_box': mbox,
            'det_ms': d['ms'], 'sam_ms': r['ms'], 'score': r['score'], 'mask': out}


def sam(image, point, out, box=None, img=None, reuse=False):
    img = imread(image) if img is None else img; pred = get_sam()
    _sync(); t0 = time.perf_counter()
    if not reuse:          # reuse=True: same image, skip re-encoding (saves about 100 ms)
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    kw = dict(point_coords=np.array([point], float), point_labels=np.array([1]), multimask_output=True)
    if box is not None:
        kw['box'] = np.array(box, float)
    masks, scores, _ = pred.predict(**kw)
    # Among the 3 candidates pick the highest score (the largest object usually covers the whole car)
    k = int(np.argmax(scores)); m = masks[k]
    _sync(); ms = 1000 * (time.perf_counter() - t0)
    np.save(out, m.astype(np.uint8))
    return {'mask': out, 'ms': ms, 'score': float(scores[k]), 'area': int(m.sum())}


def depth(model, image, mask):
    img = imread(image); H, W = img.shape[:2]
    m = np.load(mask) > 0
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); fpx = f_px(W)
    with torch.no_grad():
        if model == 'depthpro':
            from PIL import Image
            net, tf = get_depthpro()
            _sync(); t0 = time.perf_counter()
            x = tf(Image.fromarray(rgb)).to(device())
            d = net.infer(x, f_px=torch.tensor(fpx, device=device()))['depth']
        elif model == 'unidepth':
            net = get_unidepth()
            K = torch.tensor([[fpx, 0, W / 2], [0, fpx, H / 2], [0, 0, 1]], dtype=torch.float32, device=device())
            _sync(); t0 = time.perf_counter()
            x = torch.from_numpy(rgb).permute(2, 0, 1).to(device())
            d = net.infer(x, K)['depth']
        else:
            raise ValueError(f'unknown model {model}')
        _sync(); ms = 1000 * (time.perf_counter() - t0)
    d = d.squeeze().float().cpu().numpy()
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    v = d[m]; v = v[np.isfinite(v)]
    return {'median_m': float(np.median(v)), 'mean_m': float(v.mean()), 'ms': ms}


def handle(req):
    c = req.get('cmd')
    if c == 'info':
        return info()
    if c == 'sam':
        return sam(req['image'], req['point'], req['out'], req.get('box'))
    if c == 'depth':
        return depth(req['model'], req['image'], req['mask'])
    if c == 'detect':
        return detect(req['image'], req.get('conf', 0.25))
    if c == 'auto':
        return auto(req['image'], req['out'], req.get('prev_box'))
    if c == 'warmup':
        # Load the models and run one inference on a dummy image so the first real call doesn't pay CUDA kernel-compile cost
        import tempfile
        d = tempfile.mkdtemp(prefix='cadfds_warm_')
        ip, mp = os.path.join(d, 'w.png'), os.path.join(d, 'w.npy')
        cv2.imwrite(ip, np.random.default_rng(0).integers(0, 255, (1536, 2048, 3), dtype=np.uint8))
        np.save(mp, np.ones((1536, 2048), np.uint8))
        for k in req.get('models', []):
            if k == 'yolo':
                detect(ip)
            elif k == 'sam':
                sam(ip, [1024, 768], os.path.join(d, 's.npy'), [500, 400, 1500, 1100])
            else:
                depth(k, ip, mp)
        return {}
    raise ValueError(f'unknown cmd {c}')


def serve():
    out = sys.stdout
    sys.stdout = sys.stderr          # model-loading messages go to stderr so they don't corrupt the protocol
    out.write(json.dumps({'ok': True, 'ready': True}) + '\n'); out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get('cmd') == 'quit':
                break
            r = handle(req); r['ok'] = True
        except Exception as e:
            r = {'ok': False, 'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-800:]}
        r['id'] = req.get('id') if isinstance(req, dict) else None
        out.write(json.dumps(r, ensure_ascii=False) + '\n'); out.flush()


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == '--selftest':
        img, msk = sys.argv[2], sys.argv[3]
        print(json.dumps(info(), ensure_ascii=False))
        import numpy as _np
        mm = _np.load(msk); ys, xs = _np.nonzero(mm)
        pt = [float(xs.mean()), float(ys.mean())]
        tmp = os.path.join(os.environ.get('TEMP', '.'), 'cadfds_sam_selftest.npy')
        for k in range(2):
            r = sam(img, pt, tmp); print('sam', k, {a: r[a] for a in ('ms', 'score', 'area')}, 'hand-mask area', int((mm > 0).sum()),
                                     'IoU', float(((_np.load(tmp) > 0) & (mm > 0)).sum() / ((_np.load(tmp) > 0) | (mm > 0)).sum()))
        for mdl in ('unidepth', 'depthpro'):
            for k in range(2):
                print(mdl, k, depth(mdl, img, msk))
    else:
        serve()
