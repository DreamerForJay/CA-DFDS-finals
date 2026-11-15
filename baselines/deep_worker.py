# -*- coding: utf-8 -*-
"""Deep-learning comparison methods and SAM segmentation (run in a separate process with the torch environment).

Why a separate process: torch/SAM/UniDepth/Depth Pro are installed in D:\\Carla-0.10.0-Win64-Shipping\\.venv
(torch 2.11 + CUDA 12.8, supports the RTX 5080). The CA-DFDS core needs colour-science 0.4.6 in .venv-cadfds.
Keeping the two environments separate avoids package conflicts. gui_demo.py talks to this program over stdin/stdout JSON lines.

Protocol (one JSON object per line):
  {"cmd": "info"}                                          → hardware and environment info
  {"cmd": "sam", "image": p, "point": [x, y], "out": npy}  → SAM ViT-B single-point segmentation, mask saved to the out .npy
  {"cmd": "depth", "model": "unidepth"|"depthpro", "image": p, "mask": npy}
                                                           → median/mean metric depth inside the mask (m) and inference time
  {"cmd": "quit"}
Every reply carries "ok": true/false; on failure it carries "error".

Can also be run directly for testing: python deep_worker.py --selftest <image> <mask.npy>
"""
import os, sys, json, math, time, platform, traceback

BASE = os.environ.get('CADFDS_MODELS_DIR', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping')
SAM_CKPT = os.path.join(BASE, 'models', 'sam_vit_b_01ec64.pth')
DEPTHPRO_REPO = os.path.join(BASE, 'models_depthpro_repo')
UNIDEPTH_REPO = os.path.join(BASE, 'models_unidepth_repo')
UNIDEPTH_ID = 'lpiccinelli/unidepth-v2-vits14'

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
                   'unidepth': True}
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


def sam(image, point, out, box=None):
    img = imread(image); pred = get_sam()
    _sync(); t0 = time.perf_counter()
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
    if c == 'warmup':
        # Load the models and run one inference on a dummy image so the first real call doesn't pay CUDA kernel-compile cost
        import tempfile
        d = tempfile.mkdtemp(prefix='cadfds_warm_')
        ip, mp = os.path.join(d, 'w.png'), os.path.join(d, 'w.npy')
        cv2.imwrite(ip, np.random.default_rng(0).integers(0, 255, (1536, 2048, 3), dtype=np.uint8))
        np.save(mp, np.ones((1536, 2048), np.uint8))
        for k in req.get('models', []):
            if k == 'sam':
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
