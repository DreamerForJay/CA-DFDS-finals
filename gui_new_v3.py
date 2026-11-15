# -*- coding: utf-8 -*-
"""【新版 v3】CA-DFDS 車距估測 GUI（現場 demo／方法比較用）。舊版請見 gui_old_senior.py（學長舊版）。

流程：① 輸入（資料集／影像檔／影片） → ② 車輛遮罩 → ③ 估測與比較
  遮罩四種來源：
    自動偵測：YOLO11 找車 → SAM 依框分割（近距離 YOLO 認不出時改用整幅框；影片時沿用前一格的框追蹤）
    資料集遮罩：學長以 get_mask.py 手繪的遮罩（論文實驗用）
    框選＋SAM：拖一個框；可拖角點／邊調整大小、拖框內移動、滾輪縮放，SAM 即時重新分割
    手動塗抹：左鍵塗、右鍵擦
  比較 6 種方法：CA-DFDS（本研究）、CA-DFDS 重新最佳化、CA-DFDS 只用 J、灰階 DFD、UniDepth V2、ML-Depth-Pro
    （深度模型可切換「原始輸出」或「線性校正」；校正用與 CA-DFDS 相同的校正集）
CA-DFDS 在 CPU；YOLO／SAM／深度模型在 torch 環境的背景程式 baselines/deep_worker.py（GPU）。
執行：python gui_new_v3.py（或點兩下 run_gui_new.bat） [--image 影像 | --video 影片] [--mask 遮罩] [--mode auto|dataset|box|paint] [--screenshot out.png]
"""
import os, re, sys, json, time, queue, argparse, threading, subprocess, tempfile, platform, glob
import numpy as np, cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cadfds_core as C

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
TORCH_PY = os.environ.get('CADFDS_TORCH_PY', os.path.join(os.path.dirname(ROOT), '.venv', 'Scripts', 'python.exe'))
WORKER = os.path.join(ROOT, 'baselines', 'deep_worker.py')
TMP = tempfile.mkdtemp(prefix='cadfds_gui_')

ap = argparse.ArgumentParser()
ap.add_argument('--calibration', default=os.path.join(ROOT, 'results', 'calibration_methods.json'))
ap.add_argument('--image'); ap.add_argument('--video'); ap.add_argument('--mask')
ap.add_argument('--mode', choices=['auto', 'dataset', 'box', 'paint'])
ap.add_argument('--frame', type=int, default=0)
ap.add_argument('--screenshot')
ap.add_argument('--no-deep', action='store_true', help='不啟動 GPU 模型')
A = ap.parse_args()

# ---------------- 外觀 ----------------
BG, HEADER, CARD, CARD2, LINE = '#0f1626', '#0a101d', '#18223a', '#202c49', '#2c3a5e'
FG, MUTED, DIM = '#e8edf5', '#9aa8bf', '#6b7891'
ACCENT, GOOD, WARN, BAD = '#f5b700', '#5ad08a', '#ffb454', '#ff6b6b'
UI, NUM = 'Microsoft JhengHei UI', 'Segoe UI'

# key, 名稱, 顏色, 類型, 簡短說明（滑鼠移上去才顯示；主方法另有完整說明卡）
METHODS = [
    ('opt',      'CA-DFDS（本研究）',   ACCENT,    'cadfds', ''),
    ('reopt5',   'CA-DFDS 重新最佳化',  '#ff8c42', 'cadfds', '同一流程，權重改為 2026-11-15 以式 (5) 重新最佳化的 (0.9434, 0.0566, 0)'),
    ('jonly',    'CA-DFDS 只用 J',      '#e9c46a', 'cadfds', '消融對照：權重 (1, 0, 0)，只用 CIECAM02 明度 J，不用色度'),
    ('gray',     '灰階 DFD',            '#a3acb9', 'gray',   '傳統基準：灰階影像的拉普拉斯清晰度，沒有色彩資訊'),
    ('unidepth', 'UniDepth V2',         '#38bdf8', 'deep',   '他人方法：單張影像深度學習模型（ViT-S，官方權重），取遮罩內深度中位數'),
    ('depthpro', 'ML-Depth-Pro',        '#c084fc', 'deep',   '他人方法：Apple 單張影像深度學習模型（官方權重），取遮罩內深度中位數'),
]
MAIN_DESC = ('單張 f/1.4、對焦 15 m 影像 → 車輛遮罩 → CIECAM02 色彩空間（明度 J、色度 a_c、b_c）'
             '→ 各通道拉普拉斯加權 (0.8136, 0.1782, 0.0081) → 遮罩內標準差 σ → 二次校正曲線 → 距離')

CAL = json.load(open(A.calibration, encoding='utf-8')) if os.path.exists(A.calibration) else {}
if 'opt' not in CAL:
    p = os.path.join(ROOT, 'results', 'calibration_opt.json')
    w, co = C.load_calibration(p) if os.path.exists(p) else (np.array(C.W_OPT), np.array([-6.562, 128.83, 68.39]))
    CAL['opt'] = {'weights': list(w), 'coef_cm': list(co), 'r2': None}
CMP = {}
_p = os.path.join(ROOT, 'results', 'method_comparison.json')
if os.path.exists(_p):
    CMP = json.load(open(_p, encoding='utf-8'))
RMSE_HAND = CMP.get('hand', {}).get('overall_rmse_m', CMP.get('overall_rmse_m', {}))


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
        return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
    except Exception:
        return platform.processor()


def gt_of(path):
    m = re.search(r'D(\d+(?:\.\d+)?)_', os.path.basename(path or ''))
    return float(m.group(1)) if m else None


def dataset_mask_path(img_path):
    """影像 <資料集>/Txx/R/xxx.png → 依序找 <資料集>_mask、firetruck_black 共用遮罩、校正集 Training_dataset_mask。"""
    try:
        tdir = os.path.dirname(os.path.dirname(img_path)); ds = os.path.dirname(tdir)
        t = int(re.search(r'T(\d+)$', tdir).group(1))
    except Exception:
        return None, None
    dirs = [ds + '_mask', os.path.join(DATA, 'firetruck_black', os.path.basename(ds) + '_mask')]
    if 'training' in img_path.lower():
        dirs.insert(0, os.path.join(os.path.dirname(os.path.dirname(ds)), 'Training_dataset_mask'))
    return C.find_mask(dirs, t), dirs[0]


def mask_bbox(m):
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else None


# ---------------- GPU 背景程式 ----------------
class DeepWorker:
    def __init__(s, on_status):
        s.on_status, s.proc, s.info, s.lock, s.ready = on_status, None, {}, threading.Lock(), False
        threading.Thread(target=s._start, daemon=True).start()

    def _start(s):
        if A.no_deep or not os.path.exists(TORCH_PY):
            s.on_status('off', '未啟用' + ('（--no-deep）' if A.no_deep else f'（找不到 {TORCH_PY}）')); return
        s.on_status('loading', '模型載入中…')
        env = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONIOENCODING='utf-8', YOLO_VERBOSE='False')
        os.makedirs(os.path.join(ROOT, 'logs', 'scratch'), exist_ok=True)
        s.log = open(os.path.join(ROOT, 'logs', 'scratch', 'deep_worker.log'), 'w', encoding='utf-8')
        try:
            s.proc = subprocess.Popen([TORCH_PY, '-I', WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=s.log,
                                      text=True, encoding='utf-8', env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            s.proc.stdout.readline()
            s.info = s.call({'cmd': 'info'})
            s.on_status('loading', '模型載入中…')
            s.call({'cmd': 'warmup', 'models': ['yolo', 'sam', 'unidepth', 'depthpro']})
            s.ready = True; s.on_status('ready', '就緒')
        except Exception as e:
            s.on_status('off', f'啟動失敗（見 logs/scratch/deep_worker.log）：{e}')

    def call(s, req):
        with s.lock:
            s.proc.stdin.write(json.dumps(req) + '\n'); s.proc.stdin.flush()
            r = json.loads(s.proc.stdout.readline())
        if not r.get('ok'):
            raise RuntimeError(r.get('error'))
        return r

    def close(s):
        try:
            s.proc.stdin.write('{"cmd": "quit"}\n'); s.proc.stdin.flush()
        except Exception:
            pass


class Tip:
    """滑鼠移到元件上時顯示說明。"""
    def __init__(s, w, text):
        s.w, s.text, s.tw = w, text, None
        w.bind('<Enter>', s.show); w.bind('<Leave>', s.hide)

    def show(s, e=None):
        if not s.text or s.tw:
            return
        x, y = s.w.winfo_rootx() + 10, s.w.winfo_rooty() + s.w.winfo_height() + 4
        s.tw = tk.Toplevel(s.w); s.tw.wm_overrideredirect(True); s.tw.geometry(f'+{x}+{y}')
        tk.Label(s.tw, text=s.text, bg='#fff8dc', fg='#222', font=(UI, 10), justify='left', wraplength=420, padx=8, pady=5,
                 relief='solid', borderwidth=1).pack()

    def hide(s, e=None):
        if s.tw:
            s.tw.destroy(); s.tw = None


# ---------------- 主視窗 ----------------
class App:
    HANDLE = 7

    def __init__(s, root):
        s.root, s.q = root, queue.Queue()
        root.title('CA-DFDS 新版 GUI v3 — 單眼失焦車距估測・方法比較'); root.configure(bg=BG)
        st = ttk.Style(); st.theme_use('clam')
        st.configure('TCombobox', fieldbackground='#ffffff', padding=3)
        st.configure('Horizontal.TScale', background=CARD, troughcolor=LINE)
        s.img = s.mask = s.ref_mask = s.path = None
        s.box = None; s.box_drag = None; s.mask_src = ''; s.mask_info = {}; s.job = 0; s.res = {}; s.t_start = None
        s.video = None; s.playing = False; s.track_box = None; s.sam_pending = None; s.sam_busy = False
        s.src = tk.StringVar(value='dataset'); s.mode = tk.StringVar(value='dataset'); s.autorun = tk.BooleanVar(value=True)
        s.deepcal = tk.BooleanVar(value=True); s.color = tk.StringVar(); s.tidx = tk.IntVar(value=1); s.shot = tk.IntVar(value=1)
        s.brush = tk.IntVar(value=40); s.skip = tk.IntVar(value=5); s.timing = {}
        s._build(); s._hotkeys()
        s.worker = DeepWorker(lambda st_, msg: s.ui(lambda: s._worker_status(st_, msg)))
        root.after(50, s._poll); root.protocol('WM_DELETE_WINDOW', s._quit)

    # ================= 版面 =================
    def L(s, parent, text='', size=10, fg=FG, bold=False, font=UI, **kw):
        return tk.Label(parent, text=text, font=(font, size, 'bold' if bold else 'normal'), fg=fg, bg=kw.pop('bg', parent['bg']), **kw)

    def B(s, parent, text, cmd, primary=False, **kw):
        b = tk.Button(parent, text=text, command=cmd, font=(UI, 10, 'bold' if primary else 'normal'),
                      bg=ACCENT if primary else CARD2, fg='#10172a' if primary else FG,
                      activebackground='#ffd24d' if primary else LINE, activeforeground='#10172a' if primary else FG,
                      relief='flat', bd=0, padx=10, pady=5, cursor='hand2', **kw)
        return b

    def card(s, parent, title, step=None):
        f = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        f.pack(fill='x', pady=(0, 10))
        h = tk.Frame(f, bg=CARD); h.pack(fill='x', padx=12, pady=(10, 6))
        if step:
            s.L(h, f' {step} ', 10, fg='#10172a', bold=True, bg=ACCENT).pack(side='left', padx=(0, 8))
        s.L(h, title, 11, bold=True).pack(side='left')
        body = tk.Frame(f, bg=CARD); body.pack(fill='x', padx=12, pady=(0, 12))
        return body

    def radio(s, parent, var, value, text, cmd, desc=None):
        r = tk.Radiobutton(parent, text=text, value=value, variable=var, command=cmd, font=(UI, 10), bg=CARD, fg=FG,
                           selectcolor=CARD2, activebackground=CARD, activeforeground=FG, anchor='w', cursor='hand2')
        r.pack(fill='x')
        if desc:
            s.L(parent, desc, 9, fg=DIM, anchor='w', justify='left', wraplength=290).pack(fill='x', padx=(24, 0), pady=(0, 4))
        return r

    def _build(s):
        r = s.root
        # ---- 頁首 ----
        hd = tk.Frame(r, bg=HEADER, height=54); hd.pack(fill='x'); hd.pack_propagate(False)
        s.L(hd, 'CA-DFDS', 17, fg=ACCENT, bold=True, font=NUM).pack(side='left', padx=(18, 8))
        s.L(hd, '新版 v3', 10, fg='#10172a', bold=True, bg=ACCENT).pack(side='left', padx=(0, 10))
        s.L(hd, '色彩感知單眼失焦車距估測 · 方法比較 Demo', 12, fg=FG).pack(side='left')
        s.hw_gpu = s.L(hd, 'GPU 偵測中…', 9, fg=MUTED); s.hw_gpu.pack(side='right', padx=(6, 18))
        s.gpu_dot = s.L(hd, '●', 12, fg=WARN); s.gpu_dot.pack(side='right')
        s.L(hd, f'CPU  {cpu_name()}', 9, fg=MUTED).pack(side='right', padx=14)
        # ---- 主體三欄 ----
        body = tk.Frame(r, bg=BG); body.pack(fill='both', expand=True, padx=12, pady=10)
        body.columnconfigure(1, weight=1); body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=BG, width=330); left.grid(row=0, column=0, sticky='ns'); left.pack_propagate(False)
        mid = tk.Frame(body, bg=CARD, highlightbackground=LINE, highlightthickness=1); mid.grid(row=0, column=1, sticky='nsew', padx=10)
        right = tk.Frame(body, bg=BG, width=560); right.grid(row=0, column=2, sticky='ns'); right.pack_propagate(False)
        s._build_left(left); s._build_mid(mid); s._build_right(right)
        s.status = s.L(r, '', 9, fg=MUTED, anchor='w', bg=HEADER); s.status.pack(fill='x', side='bottom', ipady=4, ipadx=10)
        s._fill_colors(); s._hint()

    def _build_left(s, p):
        # ① 輸入
        b = s.card(p, '輸入', '1')
        seg = tk.Frame(b, bg=CARD); seg.pack(fill='x', pady=(0, 8))
        s.seg_btns = {}
        for v, t in [('dataset', '資料集'), ('image', '影像檔'), ('video', '影片')]:
            bt = tk.Button(seg, text=t, command=lambda v=v: s._set_src(v), font=(UI, 10), relief='flat', bd=0, pady=4, cursor='hand2')
            bt.pack(side='left', fill='x', expand=True, padx=1); s.seg_btns[v] = bt
        s.src_frames = {}
        f = tk.Frame(b, bg=CARD); s.src_frames['dataset'] = f
        g = tk.Frame(f, bg=CARD); g.pack(fill='x')
        s.L(g, '車色', 10, fg=MUTED).grid(row=0, column=0, sticky='w')
        s.cb = ttk.Combobox(g, textvariable=s.color, width=14, state='readonly', font=(UI, 10))
        s.cb.grid(row=0, column=1, sticky='w', padx=6); s.cb.bind('<<ComboboxSelected>>', lambda e: s._on_color())
        s.L(g, '張', 10, fg=MUTED).grid(row=0, column=2, padx=(8, 2))
        tk.Spinbox(g, from_=1, to=10, textvariable=s.shot, width=3, font=(UI, 10), command=s._load_from_browser).grid(row=0, column=3)
        dr = tk.Frame(f, bg=CARD); dr.pack(fill='x', pady=(8, 0))
        s.L(dr, '距離', 10, fg=MUTED).pack(side='left')
        s.tlabel = s.L(dr, '', 12, fg=ACCENT, bold=True, font=NUM); s.tlabel.pack(side='right')
        sr = tk.Frame(f, bg=CARD); sr.pack(fill='x')
        s.B(sr, '◀', lambda: s._step_t(-1), width=2).pack(side='left')
        s.scale = ttk.Scale(sr, from_=1, to=12, orient='horizontal', command=lambda v: s._on_scale(v))
        s.scale.pack(side='left', fill='x', expand=True, padx=6)
        s.B(sr, '▶', lambda: s._step_t(1), width=2).pack(side='left')
        f = tk.Frame(b, bg=CARD); s.src_frames['image'] = f
        s.B(f, '開啟影像…', s.open_img).pack(fill='x')
        s.L(f, '也可直接把影像拖進畫面。沒有資料集遮罩時會自動改用「自動偵測」。', 9, fg=DIM, wraplength=290, justify='left').pack(fill='x', pady=(6, 0))
        f = tk.Frame(b, bg=CARD); s.src_frames['video'] = f
        s.B(f, '開啟影片…', s.open_video).pack(fill='x')
        vr = tk.Frame(f, bg=CARD); vr.pack(fill='x', pady=(8, 0))
        s.play_btn = s.B(vr, '▶ 播放', s.toggle_play, primary=True); s.play_btn.pack(side='left')
        s.L(vr, '每', 10, fg=MUTED).pack(side='left', padx=(10, 2))
        tk.Spinbox(vr, from_=1, to=60, textvariable=s.skip, width=3, font=(UI, 10)).pack(side='left')
        s.L(vr, '格處理一次', 10, fg=MUTED).pack(side='left', padx=2)
        s.vscale = ttk.Scale(f, from_=0, to=1, orient='horizontal', command=lambda v: s._on_vscale(v)); s.vscale.pack(fill='x', pady=(8, 0))
        s.vlabel = s.L(f, '尚未開啟影片', 9, fg=DIM, anchor='w'); s.vlabel.pack(fill='x')
        s._set_src('dataset', load=False)
        # ② 遮罩
        b = s.card(p, '車輛遮罩（哪些像素是車）', '2')
        s.radio(b, s.mode, 'auto', '自動偵測（YOLO → SAM）', s._on_mode, '不需人工：YOLO 找車、SAM 切出輪廓；實際道路用這個')
        s.radio(b, s.mode, 'dataset', '資料集手繪遮罩', s._on_mode, '學長手繪，論文實驗使用（標準答案）')
        s.radio(b, s.mode, 'box', '框選 ＋ SAM（可調整）', s._on_mode, '拖框；拖角點／邊調大小、拖框內移動、滾輪縮放')
        s.radio(b, s.mode, 'paint', '手動塗抹', s._on_mode, '左鍵塗、右鍵擦')
        br = tk.Frame(b, bg=CARD); br.pack(fill='x', pady=(6, 0))
        s.L(br, '筆刷', 9, fg=MUTED).pack(side='left')
        ttk.Scale(br, from_=5, to=150, variable=s.brush, orient='horizontal', length=90).pack(side='left', padx=4)
        s.B(br, '指定檔…', s.open_mask).pack(side='right')
        s.B(br, '清除', s.clear).pack(side='right', padx=4)
        s.maskl = s.L(b, '', 9, fg=MUTED, anchor='w', justify='left', wraplength=300); s.maskl.pack(fill='x', pady=(8, 0))
        # ③ 估測
        b = s.card(p, '估測', '3')
        tk.Button(b, text='估測距離　Enter', command=s.run, font=(UI, 12, 'bold'), bg=ACCENT, fg='#10172a', activebackground='#ffd24d',
                  relief='flat', bd=0, pady=7, cursor='hand2').pack(fill='x')
        o = tk.Frame(b, bg=CARD); o.pack(fill='x', pady=(8, 0))
        for var, t in [(s.autorun, '換圖／改遮罩時自動估測'), (s.deepcal, '深度模型套用線性校正（公平比較）')]:
            tk.Checkbutton(o, text=t, variable=var, font=(UI, 10), bg=CARD, fg=FG, selectcolor=CARD2, activebackground=CARD,
                           activeforeground=FG, anchor='w', command=s._refresh_table).pack(fill='x')

    def _build_mid(s, p):
        top = tk.Frame(p, bg=CARD); top.pack(fill='x', padx=12, pady=(10, 4))
        s.title = s.L(top, '請選擇影像', 10, fg=MUTED, anchor='w'); s.title.pack(side='left', fill='x', expand=True)
        s.gtl = s.L(top, '', 12, fg=FG, bold=True); s.gtl.pack(side='right')
        s.cv = tk.Canvas(p, bg='#05080f', highlightthickness=0, cursor='crosshair'); s.cv.pack(fill='both', expand=True, padx=12, pady=(0, 12))
        s.cv.bind('<Configure>', lambda e: s.show())
        s.cv.bind('<ButtonPress-1>', s._press); s.cv.bind('<B1-Motion>', s._drag); s.cv.bind('<ButtonRelease-1>', s._release)
        s.cv.bind('<B3-Motion>', lambda e: s._paint(e, 0)); s.cv.bind('<Button-3>', lambda e: s._paint(e, 0))
        s.cv.bind('<MouseWheel>', s._wheel); s.cv.bind('<Motion>', s._hover)
        try:
            from tkinterdnd2 import DND_FILES
            s.cv.drop_target_register(DND_FILES); s.cv.dnd_bind('<<Drop>>', s._drop)
        except Exception:
            pass

    def _build_right(s, p):
        # 主方法結果
        b = s.card(p, 'CA-DFDS（本研究）估測結果')
        row = tk.Frame(b, bg=CARD); row.pack(fill='x')
        s.main_est = s.L(row, '—', 34, fg=ACCENT, bold=True, font=NUM); s.main_est.pack(side='left')
        r2 = tk.Frame(row, bg=CARD); r2.pack(side='left', padx=16)
        s.main_gt = s.L(r2, '真實距離 —', 11, fg=FG, anchor='w'); s.main_gt.pack(anchor='w')
        s.main_err = s.L(r2, '誤差 —', 11, fg=MUTED, anchor='w', font=NUM); s.main_err.pack(anchor='w')
        s.L(b, MAIN_DESC, 9, fg=MUTED, anchor='w', justify='left', wraplength=520).pack(fill='x', pady=(8, 0))
        # 方法比較表
        b = s.card(p, '方法比較（滑鼠移到方法名稱可看說明）')
        tb = tk.Frame(b, bg=CARD); tb.pack(fill='x')
        heads = [('', 2), ('方法', 17), ('估測', 7), ('誤差', 7), ('時間', 8), ('', 4), ('測試集 RMSE', 10)]
        for j, (h, w) in enumerate(heads):
            s.L(tb, h, 9, fg=DIM, width=w, anchor='w' if j == 1 else 'e').grid(row=0, column=j, pady=(0, 4))
        tk.Frame(tb, bg=LINE, height=1).grid(row=1, column=0, columnspan=7, sticky='ew', pady=(0, 4))
        s.cells = {}; s.group_rows = {}
        for i, (k, name, col, kind, note) in enumerate(METHODS):
            gi = i + 2 + (1 if kind != 'cadfds' else 0) + (1 if kind == 'deep' else 0)
            if kind in ('gray', 'deep') and kind not in s.group_rows:
                tk.Frame(tb, bg=LINE, height=1).grid(row=gi - 1, column=0, columnspan=7, sticky='ew', pady=3)
                s.group_rows[kind] = gi
            main = k == 'opt'
            cells = [s.L(tb, '●', 11, fg=col, width=2)]
            nm = s.L(tb, name, 10, fg=col, bold=main, width=17, anchor='w'); cells.append(nm); Tip(nm, note)
            for j in range(2, 7):
                cells.append(s.L(tb, '', 10, fg=FG, bold=main and j in (2, 3), width=heads[j][1], anchor='e', font=NUM))
            cells[5].config(text='GPU' if kind == 'deep' else 'CPU', fg=DIM, font=(NUM, 8))
            for j, c in enumerate(cells):
                c.grid(row=gi, column=j, pady=2)
            s.cells[k] = cells
        s.tbl_note = s.L(b, '', 8, fg=DIM, anchor='w', justify='left', wraplength=520); s.tbl_note.pack(fill='x', pady=(6, 0))
        # 數線
        b = s.card(p, '距離數線（白線 = 真實距離）')
        s.line = tk.Canvas(b, bg=CARD, height=142, highlightthickness=0); s.line.pack(fill='x')
        s.line.bind('<Configure>', lambda e: s._draw_line())
        # 計時
        b = s.card(p, '計時與硬體')
        s.timer = s.L(b, '', 11, fg=GOOD, bold=True, anchor='w'); s.timer.pack(fill='x')
        s.timing_f = tk.Frame(b, bg=CARD); s.timing_f.pack(fill='x', pady=(4, 0))
        s._refresh_table()

    def _hotkeys(s):
        r = s.root
        r.bind('<Return>', lambda e: s.run()); r.bind('<Left>', lambda e: s._step_t(-1)); r.bind('<Right>', lambda e: s._step_t(1))
        r.bind('<Up>', lambda e: s._step_shot(1)); r.bind('<Down>', lambda e: s._step_shot(-1)); r.bind('<space>', lambda e: s.toggle_play())

    # ================= 執行緒安全 =================
    def ui(s, fn):
        s.q.put(fn)

    def _poll(s):
        try:
            while True:
                s.q.get_nowait()()
        except queue.Empty:
            pass
        if s.t_start is not None:
            s.timer.config(text=f'⏱ 計算中… {time.perf_counter() - s.t_start:.1f} s', fg=WARN)
        s.root.after(50, s._poll)

    def _quit(s):
        s.playing = False; s.worker.close(); s.root.destroy()

    def _worker_status(s, st_, msg):
        i = s.worker.info
        g = f'GPU  {i["gpu"]}  {i["vram_gb"]} GB · CUDA {i["cuda"]} · sm_{i["sm"].replace(".", "")}' if i.get('gpu') else 'GPU'
        s.hw_gpu.config(text=f'{g}  —  {msg}')
        s.gpu_dot.config(fg={'ready': GOOD, 'loading': WARN, 'off': BAD}[st_])
        if st_ == 'ready' and s.img is not None:
            if s.mode.get() == 'auto':
                s._make_mask()
            elif s.autorun.get() and s.mask is not None and s.mask.any():
                s.run()

    def _hint(s):
        h = {'auto': '自動偵測：YOLO11 找出畫面中最大（最近）的車 → SAM 切出輪廓。近距離 YOLO 認不出時改用整幅畫面；影片中沿用前一格的框追蹤。',
             'dataset': '資料集手繪遮罩：依影像位置自動載入學長手繪的遮罩（各車色共用 firetruck_black 的遮罩）。',
             'box': '框選＋SAM：拖一個框把車框起來；拖角點或邊調整大小、拖框內移動、滾輪縮放；調整時 SAM 即時重新分割。',
             'paint': '手動塗抹：左鍵塗抹車輛範圍，右鍵擦除。'}[s.mode.get()]
        s.status.config(text=h + '　｜　←/→ 距離　↑/↓ 張數　Enter 估測　空白鍵 播放影片')

    # ================= 輸入 =================
    def _set_src(s, v, load=True):
        s.src.set(v)
        for k, bt in s.seg_btns.items():
            bt.config(bg=ACCENT if k == v else CARD2, fg='#10172a' if k == v else FG, activebackground=ACCENT if k == v else LINE)
        for k, f in s.src_frames.items():
            f.pack_forget()
        s.src_frames[v].pack(fill='x')
        if v != 'video':
            s.playing = False
        if load and v == 'image' and s.path is None:
            s.open_img()
        if load and v == 'video' and s.video is None:
            s.open_video()

    def _fill_colors(s):
        names = [os.path.basename(os.path.dirname(d)).replace('firetruck_', '') for d in sorted(glob.glob(os.path.join(DATA, 'firetruck_*', DS)))]
        names += ['校正-' + os.path.basename(os.path.dirname(d)).replace('firetruck_', '')
                  for d in sorted(glob.glob(os.path.join(DATA, 'training', 'firetruck_*', DS)))]
        s.cb['values'] = names
        if names:
            s.color.set('red' if 'red' in names else names[0]); s._on_color(load=False)

    def _ds_dir(s):
        c = s.color.get()
        return os.path.join(DATA, 'training', 'firetruck_' + c[3:], DS) if c.startswith('校正-') else os.path.join(DATA, 'firetruck_' + c, DS)

    def _on_color(s, load=True):
        s.tdirs = sorted(glob.glob(os.path.join(s._ds_dir(), 'T*')))
        s.scale.config(to=max(len(s.tdirs), 1))
        s.tidx.set(min(s.tidx.get(), len(s.tdirs)) or 1); s.scale.set(s.tidx.get()); s._tlabel()
        if load:
            s._load_from_browser()

    def _tdist(s, i):
        f = sorted(glob.glob(os.path.join(s.tdirs[i - 1], 'R', '*.png')))
        return (f, gt_of(f[0])) if f else ([], None)

    def _tlabel(s):
        if getattr(s, 'tdirs', None):
            _, d = s._tdist(s.tidx.get())
            s.tlabel.config(text=f'T{s.tidx.get():02d} · {d:.1f} m' if d else f'T{s.tidx.get():02d}')

    def _on_scale(s, v):
        i = int(round(float(v)))
        if i != s.tidx.get():
            s.tidx.set(i); s._tlabel()
            if hasattr(s, '_deb'):
                s.root.after_cancel(s._deb)
            s._deb = s.root.after(250, s._load_from_browser)

    def _step_t(s, d):
        if s.src.get() == 'video':
            return s._seek(s.video['idx'] + d * max(s.skip.get(), 1)) if s.video else None
        if getattr(s, 'tdirs', None):
            i = min(max(s.tidx.get() + d, 1), len(s.tdirs)); s.tidx.set(i); s.scale.set(i); s._tlabel(); s._load_from_browser()

    def _step_shot(s, d):
        s.shot.set(min(max(s.shot.get() + d, 1), 10)); s._load_from_browser()

    def _load_from_browser(s):
        if getattr(s, 'tdirs', None):
            files, _ = s._tdist(s.tidx.get())
            if files:
                s._set_src('dataset', load=False); s.load_image(files[min(s.shot.get(), len(files)) - 1])

    def _sync_browser(s, p):
        m = re.search(r'(training[\\/])?firetruck_([^\\/]+)[\\/][^\\/]+[\\/]T(\d+)[\\/]R[\\/]', p, re.I)
        if not m:
            return False
        name = ('校正-' if m.group(1) else '') + m.group(2)
        if name in s.cb['values'] and name != s.color.get():
            s.color.set(name); s._on_color(load=False)
        s.tidx.set(int(m.group(3))); s.scale.set(int(m.group(3))); s._tlabel()
        files, _ = s._tdist(s.tidx.get()) if getattr(s, 'tdirs', None) else ([], None)
        norm = [os.path.normcase(os.path.abspath(f)) for f in files]
        if os.path.normcase(os.path.abspath(p)) in norm:
            s.shot.set(norm.index(os.path.normcase(os.path.abspath(p))) + 1)
        return True

    def open_img(s):
        p = filedialog.askopenfilename(initialdir=os.path.dirname(s.path) if s.path else DATA, filetypes=[('Image', '*.png *.jpg *.jpeg *.bmp')])
        if p:
            s.load_image(p)

    def open_video(s):
        p = filedialog.askopenfilename(initialdir=ROOT, filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('All', '*.*')])
        if p:
            s.load_video(p)

    def open_mask(s):
        if s.img is None:
            return
        _, d = dataset_mask_path(s.path) if s.path else (None, DATA)
        p = filedialog.askopenfilename(initialdir=d if d and os.path.isdir(d) else DATA, filetypes=[('Mask', '*.npy *.png')])
        if p:
            s.mode.set('paint'); s.set_mask(C.load_mask(p, s.img.shape), f'指定檔案 {os.path.basename(p)}'); s._maybe_run()

    def _drop(s, e):
        for p in s.root.tk.splitlist(e.data):
            low = p.lower()
            if low.endswith('.npy') or '_mask' in low:
                if s.img is not None:
                    s.mode.set('paint'); s.set_mask(C.load_mask(p, s.img.shape), f'拖入 {os.path.basename(p)}'); s._maybe_run()
            elif low.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                s.load_video(p); return
            else:
                s.load_image(p); return

    def _new_frame(s, img, path, title):
        s.img, s.path = img, path; s.job += 1; s.res = {}; s.t_start = None; s.timing = {}
        s.timer.config(text=''); s._show_timing()
        s.title.config(text=title)
        g = gt_of(path); s.gtl.config(text=f'真實距離 {g:.2f} m' if g else '')
        mp, _ = dataset_mask_path(path) if path else (None, None)
        s.ref_mask = C.load_mask(mp, img.shape) if mp else None; s.ref_mask_path = mp

    def load_image(s, p):
        img = C.imread(p)
        if img is None:
            messagebox.showerror('CA-DFDS', f'無法讀取影像：{p}'); return
        s.video = None; s.playing = False
        in_ds = s._sync_browser(p)
        if not in_ds:
            s._set_src('image', load=False)
        rel = os.path.relpath(p, DATA) if p.lower().startswith(DATA.lower()) else p
        s._new_frame(img, p, rel)
        if s.mode.get() == 'dataset' and s.ref_mask is None:
            s.mode.set('auto'); s._hint()
        s._clear_results(); s._make_mask()

    def load_video(s, p):
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            messagebox.showerror('CA-DFDS', f'無法開啟影片：{p}'); return
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        s.video = {'cap': cap, 'n': n, 'fps': fps, 'idx': 0, 'name': os.path.basename(p)}; s.track_box = None
        s._set_src('video', load=False); s.vscale.config(to=max(n - 1, 1))
        if s.mode.get() == 'dataset':
            s.mode.set('auto'); s._hint()
        s._seek(A.frame if A.video else 0)

    def _seek(s, i):
        v = s.video
        if not v:
            return
        i = int(min(max(i, 0), v['n'] - 1)); v['cap'].set(cv2.CAP_PROP_POS_FRAMES, i); ok, frame = v['cap'].read()
        if not ok:
            s.playing = False; s.play_btn.config(text='▶ 播放'); return
        v['idx'] = i; s.vscale.set(i)
        s.vlabel.config(text=f'第 {i + 1} / {v["n"]} 格　{i / v["fps"]:.2f} s　（{v["fps"]:.0f} fps）')
        s._new_frame(frame, None, f'影片 {v["name"]}　第 {i + 1} 格')
        s._clear_results(); s._make_mask()

    def _on_vscale(s, v):
        if s.video and int(float(v)) != s.video['idx'] and not s.playing:
            if hasattr(s, '_vdeb'):
                s.root.after_cancel(s._vdeb)
            s._vdeb = s.root.after(200, lambda: s._seek(int(float(v))))

    def toggle_play(s):
        if not s.video:
            return
        s.playing = not s.playing; s.play_btn.config(text='⏸ 暫停' if s.playing else '▶ 播放')
        if s.playing and s.t_start is None:
            s._next_frame()

    def _next_frame(s):
        if s.playing and s.video:
            if s.video['idx'] + s.skip.get() >= s.video['n']:
                s.playing = False; s.play_btn.config(text='▶ 播放'); return
            s._seek(s.video['idx'] + max(s.skip.get(), 1))

    def _img_ref(s):
        """給 GPU 背景程式的影像路徑；影片格存成 .npy（免 PNG 編碼）。"""
        if s.path:
            return s.path
        p = os.path.join(TMP, f'frame_{s.job}.npy'); np.save(p, s.img); return p

    # ================= 遮罩 =================
    def set_mask(s, m, src, info=None):
        s.mask = m.astype(bool); s.mask_src = src; s.mask_info = info or {}
        if s.mode.get() in ('auto', 'dataset') or s.box is None:
            s.box = mask_bbox(s.mask) if s.mode.get() != 'paint' else None
        txt = f'來源：{src}'
        if s.mask.any():
            txt += f'\n面積 {100 * s.mask.mean():.1f}% 畫面'
            if s.ref_mask is not None and s.mode.get() != 'dataset':
                iou = (s.mask & s.ref_mask).sum() / max((s.mask | s.ref_mask).sum(), 1)
                txt += f'　與手繪遮罩 IoU = {iou:.2f}'
        s.maskl.config(text=txt); s.show()

    def clear(s):
        if s.img is not None:
            s.box = None; s.set_mask(np.zeros(s.img.shape[:2], bool), '已清除'); s._clear_results()

    def _on_mode(s):
        s._hint()
        if s.img is not None and s.mode.get() in ('auto', 'dataset'):
            s._make_mask()
        else:
            s.show()

    def _make_mask(s):
        """依目前模式產生遮罩，完成後（若勾選）自動估測。"""
        if s.img is None:
            return
        m = s.mode.get()
        if m == 'dataset':
            if s.ref_mask is not None:
                s.set_mask(s.ref_mask.copy(), f'資料集手繪遮罩 {os.path.basename(s.ref_mask_path)}'); s._maybe_run()
            else:
                s.set_mask(np.zeros(s.img.shape[:2], bool), '此影像沒有資料集遮罩 → 請改用「自動偵測」')
        elif m == 'auto':
            if not s.worker.ready:
                s.set_mask(np.zeros(s.img.shape[:2], bool), '等待 GPU 模型載入（完成後自動偵測）'); return
            s._auto_mask()
        else:
            if s.mask is None or s.mask.shape != s.img.shape[:2]:
                s.mask = np.zeros(s.img.shape[:2], bool)
            if m == 'box' and s.box:
                s._request_sam(s.box, run_after=True)
            else:
                s.set_mask(s.mask, s.mask_src or '手動'); s._maybe_run()

    def _auto_mask(s):
        job, ref, out = s.job, s._img_ref(), os.path.join(TMP, f'auto_{s.job}.npy')
        prev = s.track_box if s.video else None
        s.maskl.config(text='來源：自動偵測中（YOLO → SAM）…')

        def work():
            try:
                r = s.worker.call({'cmd': 'auto', 'image': ref, 'out': out, 'prev_box': prev})
                m = np.load(out) > 0
                src_t = {'yolo': 'YOLO 偵測', 'track': '沿用前一格追蹤', 'fallback': '近距離整幅畫面（YOLO 未偵測到）'}[r['source']]
                det = r['dets'][0] if r['dets'] else None
                info = {'det_ms': r['det_ms'], 'sam_ms': r['sam_ms']}
                txt = f'自動：{src_t}' + (f'（{det["cls"]} {det["conf"]:.2f}）' if det and r['source'] == 'yolo' else '')

                def done():
                    if job != s.job:
                        return
                    s.timing['遮罩'] = f'YOLO {r["det_ms"]:.0f} ms + SAM {r["sam_ms"]:.0f} ms（GPU）'
                    s.track_box = r['mask_box']; s.box = r['box'] if r['source'] == 'yolo' else mask_bbox(m)
                    s.set_mask(m, txt, info); s.box = r['box'] if r['source'] == 'yolo' else mask_bbox(m); s.show(); s._maybe_run()
                s.ui(done)
            except Exception as ex:
                s.ui(lambda: s.maskl.config(text=f'自動偵測失敗：{ex}'))
        threading.Thread(target=work, daemon=True).start()

    def _request_sam(s, box, run_after=False):
        """框選模式：只保留最新的框請求（拖曳時不會堆積）。"""
        if not s.worker.ready:
            s.maskl.config(text='GPU 模型尚未就緒，無法使用 SAM'); return
        s.sam_pending = (list(box), run_after, s.job)
        if not s.sam_busy:
            s.sam_busy = True; threading.Thread(target=s._sam_loop, daemon=True).start()

    def _sam_loop(s):
        while s.sam_pending:
            box, run_after, job = s.sam_pending; s.sam_pending = None
            out = os.path.join(TMP, 'sam.npy')
            try:
                r = s.worker.call({'cmd': 'sam', 'image': s._img_ref(), 'point': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                                   'box': box, 'out': out})
                m = np.load(out) > 0

                def done(m=m, r=r, run_after=run_after, job=job):
                    if job != s.job:
                        return
                    s.timing['遮罩'] = f'SAM {r["ms"]:.0f} ms（GPU，框選）'
                    s.set_mask(m, f'框選＋SAM（分數 {r["score"]:.2f}）')
                    if run_after and not s.sam_pending:
                        s._maybe_run()
                s.ui(done)
            except Exception as ex:
                s.ui(lambda ex=ex: s.maskl.config(text=f'SAM 失敗：{ex}'))
        s.sam_busy = False

    # ================= 滑鼠：框編輯 =================
    def _xy(s, e):
        return (e.x - s.ox) / s.sc, (e.y - s.oy) / s.sc

    def _hit(s, e):
        """回傳框上的控制點代號：'nw','n',...,'move' 或 None。"""
        if not s.box:
            return None
        x0, y0, x1, y1 = [v * s.sc + o for v, o in zip(s.box, (s.ox, s.oy, s.ox, s.oy))]
        H = s.HANDLE + 3; xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        for name, (hx, hy) in {'nw': (x0, y0), 'ne': (x1, y0), 'sw': (x0, y1), 'se': (x1, y1),
                               'n': (xm, y0), 's': (xm, y1), 'w': (x0, ym), 'e': (x1, ym)}.items():
            if abs(e.x - hx) <= H and abs(e.y - hy) <= H:
                return name
        if x0 < e.x < x1 and y0 < e.y < y1:
            return 'move'
        return None

    def _hover(s, e):
        if s.img is None or s.mode.get() not in ('box', 'auto'):
            return
        h = s._hit(e)
        s.cv.config(cursor={'nw': 'size_nw_se', 'se': 'size_nw_se', 'ne': 'size_ne_sw', 'sw': 'size_ne_sw', 'n': 'sb_v_double_arrow',
                            's': 'sb_v_double_arrow', 'w': 'sb_h_double_arrow', 'e': 'sb_h_double_arrow', 'move': 'fleur'}.get(h, 'crosshair'))

    def _press(s, e):
        if s.img is None:
            return
        m = s.mode.get()
        if m in ('box', 'auto', 'dataset'):
            h = s._hit(e)
            if m != 'box':           # 在自動／資料集遮罩上拖動 → 轉為可調整的框選模式
                if not h:
                    return
                s.mode.set('box'); s._hint()
            x, y = s._xy(e)
            s.box_drag = (h or 'new', x, y, list(s.box) if s.box else None)
            if not h:
                s.box = [x, y, x, y]
        else:
            s._paint(e, 1)

    def _drag(s, e):
        if s.img is None:
            return
        if s.mode.get() == 'box' and s.box_drag:
            h, x0, y0, b0 = s.box_drag; x, y = s._xy(e); W, H = s.img.shape[1], s.img.shape[0]
            if h == 'new':
                b = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            elif h == 'move':
                dx, dy = x - x0, y - y0; bw, bh = b0[2] - b0[0], b0[3] - b0[1]
                nx, ny = min(max(b0[0] + dx, 0), W - bw), min(max(b0[1] + dy, 0), H - bh); b = [nx, ny, nx + bw, ny + bh]
            else:
                b = list(b0)
                if 'w' in h: b[0] = min(x, b[2] - 10)
                if 'e' in h: b[2] = max(x, b[0] + 10)
                if 'n' in h: b[1] = min(y, b[3] - 10)
                if 's' in h: b[3] = max(y, b[1] + 10)
            s.box = [min(max(b[0], 0), W), min(max(b[1], 0), H), min(max(b[2], 0), W), min(max(b[3], 0), H)]
            s.show(fast=True)
            if s.box[2] - s.box[0] > 20 and s.box[3] - s.box[1] > 20:
                if hasattr(s, '_sdeb'):
                    s.root.after_cancel(s._sdeb)
                s._sdeb = s.root.after(120, lambda: s._request_sam(s.box))   # 拖曳中即時預覽分割
        elif s.mode.get() == 'paint':
            s._paint(e, 1)

    def _release(s, e):
        if s.img is None:
            return
        if s.mode.get() == 'box' and s.box_drag:
            s.box_drag = None
            if s.box and s.box[2] - s.box[0] > 20 and s.box[3] - s.box[1] > 20:
                s.job += 1; s._request_sam(s.box, run_after=True)
            else:
                s.box = None; s.show()
        elif s.mode.get() == 'paint':
            s.job += 1; s._maybe_run()

    def _wheel(s, e):
        """滾輪：以框中心等比例縮放框，SAM 重新分割並估測。"""
        if s.img is None or not s.box:
            return
        if s.mode.get() != 'box':
            s.mode.set('box'); s._hint()
        f = 1.06 if e.delta > 0 else 1 / 1.06; W, H = s.img.shape[1], s.img.shape[0]
        cx, cy = (s.box[0] + s.box[2]) / 2, (s.box[1] + s.box[3]) / 2; hw, hh = (s.box[2] - s.box[0]) / 2 * f, (s.box[3] - s.box[1]) / 2 * f
        s.box = [max(cx - hw, 0), max(cy - hh, 0), min(cx + hw, W), min(cy + hh, H)]; s.show(fast=True)
        if hasattr(s, '_wdeb'):
            s.root.after_cancel(s._wdeb)
        s._wdeb = s.root.after(300, lambda: (setattr(s, 'job', s.job + 1), s._request_sam(s.box, run_after=True)))

    def _paint(s, e, v):
        if s.img is None:
            return
        if s.mode.get() != 'paint':
            s.mode.set('paint'); s._hint()
        x, y = s._xy(e); m = s.mask.astype(np.uint8)
        cv2.circle(m, (int(x), int(y)), int(s.brush.get() / s.sc), int(v), -1)
        s.mask = m > 0; s.mask_src = '手動塗抹'; s.box = None; s.show(fast=True)

    # ================= 繪圖 =================
    def show(s, fast=False):
        c = s.cv; c.delete('all')
        W, H = max(c.winfo_width(), 100), max(c.winfo_height(), 100)
        if s.img is None:
            c.create_text(W / 2, H / 2, text='① 從左側選擇資料集影像、開啟影像檔或影片\n（也可以把檔案拖進來）', fill=MUTED, font=(UI, 15), justify='center')
            return
        s.sc = min(W / s.img.shape[1], H / s.img.shape[0])
        nw, nh = int(s.img.shape[1] * s.sc), int(s.img.shape[0] * s.sc); s.ox, s.oy = (W - nw) // 2, (H - nh) // 2
        d = cv2.resize(s.img, (nw, nh), interpolation=cv2.INTER_NEAREST if fast else cv2.INTER_AREA)
        if s.mask is not None and s.mask.shape == s.img.shape[:2]:
            m = cv2.resize(s.mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
            if m.any():
                d[~m] = (0.45 * d[~m]).astype(np.uint8)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(d, cs, -1, (138, 208, 90), 2)
        s.tkimg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)))
        c.create_image(s.ox, s.oy, anchor='nw', image=s.tkimg)
        if s.box:
            x0, y0, x1, y1 = [v * s.sc + o for v, o in zip(s.box, (s.ox, s.oy, s.ox, s.oy))]
            editable = s.mode.get() == 'box'
            c.create_rectangle(x0, y0, x1, y1, outline=ACCENT, width=2, dash=() if editable else (6, 4))
            if editable or s.mode.get() == 'auto':
                xm, ym, h = (x0 + x1) / 2, (y0 + y1) / 2, s.HANDLE
                for hx, hy in [(x0, y0), (x1, y0), (x0, y1), (x1, y1), (xm, y0), (xm, y1), (x0, ym), (x1, ym)]:
                    c.create_rectangle(hx - h / 2, hy - h / 2, hx + h / 2, hy + h / 2, fill=ACCENT, outline='#10172a')
            s._box_label(x0, y0, x1, y1)

    def _box_label(s, x0, y0, x1, y1):
        r = s.res.get('opt')
        if not r or 'est' not in r:
            return
        g = gt_of(s.path); txt = f'CA-DFDS  {r["est"]:.2f} m' + (f'   誤差 {r["est"] - g:+.2f}' if g else '')
        tw = 9 * len(txt) + 20; bx = min(max(x0, 4), s.cv.winfo_width() - tw - 4); by = y0 - 30 if y0 > 34 else y0 + 4
        s.cv.create_rectangle(bx, by, bx + tw, by + 26, fill=ACCENT, outline='')
        s.cv.create_text(bx + 10, by + 13, text=txt, fill='#10172a', anchor='w', font=(UI, 11, 'bold'))

    def _est(s, k):
        """回傳要顯示的估測值：深度模型依設定套用線性校正。"""
        r = s.res.get(k, {})
        if 'est' not in r:
            return None
        if k in ('unidepth', 'depthpro') and s.deepcal.get() and k + '_lin' in CAL:
            return float(np.polyval(CAL[k + '_lin']['coef_cm'], r['est'])) / 100
        return r['est']

    def _draw_line(s):
        c = s.line; c.delete('all'); W = c.winfo_width(); g = gt_of(s.path)
        vals = [v for v in (s._est(k) for k, *_ in METHODS) if v is not None] + ([g] if g else [])
        lo, hi = 0.0, max(8.0, np.ceil(max(vals)) if vals else 8.0)
        L, R = 130, W - 24
        X = lambda v: L + (min(max(v, lo), hi) - lo) / (hi - lo) * (R - L)
        ya = 128
        c.create_line(L, ya, R, ya, fill=DIM)
        for v in range(int(lo), int(hi) + 1):
            c.create_line(X(v), ya - 3, X(v), ya + 3, fill=DIM); c.create_text(X(v), ya + 12, text=f'{v} m', fill=DIM, font=(NUM, 8))
        if g:
            c.create_line(X(g), 2, X(g), ya, fill='#ffffff', width=2, dash=(4, 2))
        for i, (k, name, col, *_) in enumerate(METHODS):
            y = 10 + i * 19
            c.create_text(L - 8, y, text=name, fill=col, anchor='e', font=(UI, 9, 'bold' if k == 'opt' else 'normal'))
            e = s._est(k)
            if e is None:
                continue
            x = X(e)
            if g:
                c.create_line(X(g), y, x, y, fill=col, width=3)
            rad = 7 if k == 'opt' else 5
            c.create_oval(x - rad, y - rad, x + rad, y + rad, fill=col, outline='#ffffff' if k == 'opt' else col, width=2)
            right = (e >= g if g else True) and x < R - 50     # left of the true value → label goes on the left so it doesn't cover the white line
            c.create_text(x + (11 if right else -11), y, text=f'{e:.2f}', fill=col, anchor='w' if right else 'e', font=(NUM, 9, 'bold'))

    # ================= 結果 =================
    def _clear_results(s):
        s.res = {}; s._refresh_table(); s.show()

    def _refresh_table(s):
        if not hasattr(s, 'cells'):
            return
        g = gt_of(s.path); errs = {}
        for k, row in s.cells.items():
            e = s._est(k); r = s.res.get(k, {})
            rk = k + '_lin' if k in ('unidepth', 'depthpro') and s.deepcal.get() and k + '_lin' in CAL else k
            rm = RMSE_HAND.get(rk)
            row[6].config(text=f'{rm:.3f} m' if rm is not None else '—', fg=MUTED)
            if 'err' in r:
                row[2].config(text='失敗'); row[3].config(text=''); row[4].config(text=''); continue
            row[2].config(text=f'{e:.2f} m' if e is not None else '')
            row[4].config(text=f'{r["ms"]:.0f} ms' if 'ms' in r else '')
            if e is not None and g:
                d = e - g; errs[k] = abs(d)
                row[3].config(text=f'{d:+.2f} m', fg=GOOD if abs(d) < 0.2 else WARN if abs(d) < 0.5 else BAD)
            else:
                row[3].config(text='', fg=FG)
        best = min(errs, key=errs.get) if errs else None
        for k, row in s.cells.items():
            row[0].config(text='★' if k == best else '●')
        e = s._est('opt')
        s.main_est.config(text=f'{e:.2f} m' if e is not None else '—')
        s.main_gt.config(text=f'真實距離 {g:.2f} m' if g else '真實距離 —（無標註）')
        if e is not None and g:
            d = e - g; s.main_err.config(text=f'誤差 {d:+.2f} m（{100 * abs(d) / g:.1f}%）', fg=GOOD if abs(d) < 0.2 else WARN if abs(d) < 0.5 else BAD)
        else:
            s.main_err.config(text='誤差 —', fg=MUTED)
        mode = '線性校正' if s.deepcal.get() else '原始輸出（未校正）'
        s.tbl_note.config(text=f'誤差 = 估測 − 真實；★ = 本張最接近真值。深度模型目前顯示：{mode}。'
                               f'測試集 RMSE = 紅綠藍黑 4 色 × 1.2–6.7 m 共 480 張（手繪遮罩）。')
        s._draw_line()

    def _set_result(s, job, k, **r):
        if job != s.job:
            return
        s.res[k] = r
        if 'ms' in r:
            nm = {'opt': 'CA-DFDS（3 種權重共用）', 'gray': '灰階 DFD', 'unidepth': 'UniDepth V2', 'depthpro': 'ML-Depth-Pro'}.get(k)
            if nm:
                s.timing[nm] = f'{r["ms"]:.0f} ms（{"GPU" if k in ("unidepth", "depthpro") else "CPU"}）'
        s._refresh_table(); s.show(); s._show_timing()

    def _show_timing(s):
        for w in s.timing_f.winfo_children():
            w.destroy()
        for i, (k, v) in enumerate(s.timing.items()):     # two columns, so the card doesn't get too tall
            r_, c_ = divmod(i, 2)
            s.L(s.timing_f, k.replace('（3 種權重共用）', ''), 9, fg=MUTED, anchor='w').grid(row=r_, column=2 * c_, sticky='w', padx=(0, 8))
            s.L(s.timing_f, v, 9, fg=FG, anchor='w').grid(row=r_, column=2 * c_ + 1, sticky='w', padx=(0, 18))

    def _maybe_run(s):
        if s.autorun.get() and s.img is not None and s.mask is not None and s.mask.sum() >= 100:
            s.run()

    def run(s):
        if s.img is None or s.mask is None or s.mask.sum() < 100:
            if s.playing:
                s.root.after(30, s._next_frame)       # 這格沒有可用遮罩 → 跳下一格
            else:
                messagebox.showinfo('CA-DFDS', '請先選影像並產生車輛遮罩（② 的任一方式）')
            return
        s.job += 1; job = s.job; s.res = {}; s._refresh_table(); s.t_start = time.perf_counter()
        img, mask = s.img, s.mask.copy(); ref = s._img_ref()
        for k in [k for k in s.timing if k != '遮罩']:
            s.timing.pop(k)
        pending = {'cpu'}
        deep = [m[0] for m in METHODS if m[3] == 'deep'] if s.worker.ready else []
        if deep:
            pending.add('gpu')

        def finish(part):
            pending.discard(part)
            if not pending and job == s.job:
                tot = time.perf_counter() - s.t_start; s.t_start = None
                s.timer.config(text=f'⏱ 完成，總時間 {tot:.2f} s（CPU 與 GPU 平行）', fg=GOOD)
                if s.playing:
                    s.root.after(30, s._next_frame)

        def cpu():
            ci, cm = C.crop_to_mask(img, mask)
            t0 = time.perf_counter(); gs = C.gray_sigma(ci, cm); tg = 1000 * (time.perf_counter() - t0)
            if 'gray' in CAL:
                s.ui(lambda: s._set_result(job, 'gray', est=np.polyval(CAL['gray']['coef_cm'], gs) / 100, ms=tg))
            t0 = time.perf_counter(); cov = C.laplacian_cov(ci, cm); tc = 1000 * (time.perf_counter() - t0)
            for k in ('opt', 'reopt5', 'jonly'):
                if k in CAL:
                    sg = C.sigma_eq5(cov, CAL[k]['weights'])
                    s.ui(lambda k=k, sg=sg: s._set_result(job, k, est=np.polyval(CAL[k]['coef_cm'], sg) / 100, ms=tc))
            s.ui(lambda: finish('cpu'))

        def gpu():
            mp = os.path.join(TMP, f'mask_{job}.npy'); np.save(mp, mask.astype(np.uint8))
            for k in deep:
                if job != s.job:
                    return
                try:
                    r = s.worker.call({'cmd': 'depth', 'model': k, 'image': ref, 'mask': mp})
                    s.ui(lambda k=k, r=r: s._set_result(job, k, est=r['median_m'], ms=r['ms']))
                except Exception as ex:
                    s.ui(lambda k=k, ex=ex: s._set_result(job, k, err=str(ex)))
            s.ui(lambda: finish('gpu'))
        threading.Thread(target=cpu, daemon=True).start()
        if deep:
            threading.Thread(target=gpu, daemon=True).start()


def grab_window(root):
    """只擷取本視窗內容（Windows PrintWindow），不會拍到疊在上面的其他視窗或通知。"""
    from PIL import ImageGrab
    try:
        import ctypes
        from ctypes import wintypes
        u, g = ctypes.windll.user32, ctypes.windll.gdi32
        hwnd = u.GetParent(root.winfo_id()) or root.winfo_id()
        r = wintypes.RECT(); u.GetClientRect(hwnd, ctypes.byref(r)); w, h = r.right, r.bottom
        hdc = u.GetDC(hwnd); mdc = g.CreateCompatibleDC(hdc); bmp = g.CreateCompatibleBitmap(hdc, w, h); g.SelectObject(mdc, bmp)
        u.PrintWindow(hwnd, mdc, 3)

        class BIH(ctypes.Structure):
            _fields_ = [('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG), ('biPlanes', wintypes.WORD),
                        ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD), ('biSizeImage', wintypes.DWORD),
                        ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG), ('biClrUsed', wintypes.DWORD),
                        ('biClrImportant', wintypes.DWORD)]
        bi = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0); buf = ctypes.create_string_buffer(w * h * 4)
        g.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
        g.DeleteObject(bmp); g.DeleteDC(mdc); u.ReleaseDC(hwnd, hdc)
        return Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1).convert('RGB')
    except Exception:
        x, y = root.winfo_rootx(), root.winfo_rooty()
        return ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height()), all_screens=True)


def main():
    try:
        import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        from tkinterdnd2 import TkinterDnD; root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    root.geometry('1880x1020'); root.minsize(1500, 860)
    try:
        root.state('zoomed')
    except Exception:
        pass
    app = App(root)
    if A.mode:
        app.mode.set(A.mode); app._hint()

    def start():
        if A.video:
            app.load_video(A.video)
        elif A.image:
            app.load_image(A.image)
            if A.mask:
                app.mode.set('paint'); app.set_mask(C.load_mask(A.mask, app.img.shape), os.path.basename(A.mask)); app._maybe_run()
    root.after(300, start)
    if A.screenshot:
        def snap(n=0):
            need = {m[0] for m in METHODS if m[3] != 'deep' or not A.no_deep}
            busy = (not app.worker.ready and not A.no_deep) or app.t_start is not None or not need <= set(app.res)
            if busy and n < 300:
                root.after(500, lambda: snap(n + 1)); return
            root.update(); grab_window(root).save(A.screenshot); app._quit()
        root.after(1500, snap)
    root.mainloop()


if __name__ == '__main__':
    main()
