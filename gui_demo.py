# -*- coding: utf-8 -*-
"""CA-DFDS 車距估測 GUI v2（現場 demo／方法比較用）。

一張影像同時比較 6 種方法：
  CA-DFDS（本研究，論文權重）／CA-DFDS 重新最佳化／CA-DFDS J only／灰階 DFD（無色彩基準）
  ＋ UniDepth V2、ML-Depth-Pro（單張影像深度學習模型，GPU）
操作流程（詳見 docs/GUI_DEMO.md）：
  1. 選影像：上方「車色／距離滑桿／張數」瀏覽資料集，或「開啟影像…」，或把檔案拖進畫面。
  2. 遮罩自動處理：預設自動載入資料集的手繪遮罩；也可切換「SAM 框選」用滑鼠拖一個框自動分割，或「手動塗抹」。
  3. 勾選「自動估測」時換圖即計算；結果表、距離數線與影像上的標籤會隨各方法完成即時更新。
CA-DFDS 在 CPU 計算；深度模型與 SAM 在另一個 torch 環境（GPU）的背景程式 baselines/deep_worker.py 執行。
執行：python gui_demo.py [--image 影像] [--mask 遮罩] [--screenshot out.png]
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
ap.add_argument('--image'); ap.add_argument('--mask'); ap.add_argument('--screenshot')
ap.add_argument('--no-deep', action='store_true', help='不啟動 GPU 深度模型')
A = ap.parse_args()

# ---------------- 方法與校正 ----------------
BG, PANEL, FG, MUTED = '#141b2d', '#1d2740', '#e8edf5', '#93a1b8'
FONT = 'Microsoft JhengHei UI'
METHODS = [  # key, 顯示名稱, 顏色, 類型, 說明
    ('opt',      'CA-DFDS（本研究）',  '#f5b700', 'cadfds', 'CIECAM02 J/a_c/b_c 權重 (0.8136, 0.1782, 0.0081)'),
    ('reopt5',   'CA-DFDS 重新最佳化', '#ff8c42', 'cadfds', '式 (5) 重新最佳化權重'),
    ('jonly',    'CA-DFDS J only',     '#ffd98e', 'cadfds', '只用明度 J'),
    ('gray',     '灰階 DFD（無色彩）', '#9aa5b1', 'gray',   '傳統灰階拉普拉斯清晰度'),
    ('unidepth', 'UniDepth V2',        '#38bdf8', 'deep',   '單張深度模型，官方權重，不校正'),
    ('depthpro', 'ML-Depth-Pro',       '#c084fc', 'deep',   '單張深度模型，官方權重，不校正'),
]
MCOL = {m[0]: m[2] for m in METHODS}
CAL = {}
if os.path.exists(A.calibration):
    CAL = json.load(open(A.calibration, encoding='utf-8'))
if 'opt' not in CAL:   # 舊版相容：只有 calibration_opt.json 或論文曲線
    p = os.path.join(ROOT, 'results', 'calibration_opt.json')
    w, co = C.load_calibration(p) if os.path.exists(p) else (np.array(C.W_OPT), np.array([-6.562, 128.83, 68.39]))
    CAL['opt'] = {'weights': list(w), 'coef_cm': list(co), 'r2': None}
RMSE = {}   # 測試集整體 RMSE（results/method_comparison.json，由 scripts/compare_methods.py 產生）
_p = os.path.join(ROOT, 'results', 'method_comparison.json')
if os.path.exists(_p):
    RMSE = json.load(open(_p, encoding='utf-8')).get('overall_rmse_m', {})


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


def auto_mask_path(img_path):
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


# ---------------- GPU 背景程式 ----------------
class DeepWorker:
    def __init__(s, on_status):
        s.on_status, s.proc, s.info, s.lock, s.ready = on_status, None, {}, threading.Lock(), False
        threading.Thread(target=s._start, daemon=True).start()

    def _start(s):
        if A.no_deep or not os.path.exists(TORCH_PY):
            s.on_status('off', f'未啟用（找不到 {TORCH_PY}）' if not A.no_deep else '未啟用（--no-deep）'); return
        s.on_status('loading', 'GPU 模型載入中…（約 20–40 秒）')
        env = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONIOENCODING='utf-8')
        os.makedirs(os.path.join(ROOT, 'logs', 'scratch'), exist_ok=True)
        s.log = open(os.path.join(ROOT, 'logs', 'scratch', 'deep_worker.log'), 'w', encoding='utf-8')
        try:
            s.proc = subprocess.Popen([TORCH_PY, '-I', WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=s.log,
                                      text=True, encoding='utf-8', env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            s.proc.stdout.readline()
            s.info = s.call({'cmd': 'info'})
            s.call({'cmd': 'warmup', 'models': ['sam', 'unidepth', 'depthpro']})
            s.ready = True; s.on_status('ready', '就緒')
        except Exception as e:
            s.on_status('off', f'啟動失敗：{e}（見 logs/scratch/deep_worker.log）')

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


# ---------------- 主視窗 ----------------
class App:
    def __init__(s, root):
        s.root, s.q = root, queue.Queue()
        root.title('CA-DFDS 單眼失焦車距估測 — 方法比較 Demo'); root.configure(bg=BG)
        st = ttk.Style(); st.theme_use('clam')
        st.configure('TCombobox', fieldbackground='#ffffff'); st.configure('Horizontal.TScale', background=BG)
        s.img = s.mask = s.ref_mask = s.path = None; s.mask_src = ''; s.job = 0; s.res = {}; s.t_start = None
        s.mode = tk.StringVar(value='auto'); s.autorun = tk.BooleanVar(value=True)
        s.color = tk.StringVar(); s.tidx = tk.IntVar(value=1); s.shot = tk.IntVar(value=1); s.brush = tk.IntVar(value=30)
        s._build(); s._hotkeys()
        s.worker = DeepWorker(lambda st_, msg: s.ui(lambda: s._worker_status(st_, msg)))
        root.after(50, s._poll); root.protocol('WM_DELETE_WINDOW', s._quit)

    # ---- 版面 ----
    def _btn(s, parent, text, cmd, **kw):
        return tk.Button(parent, text=text, command=cmd, font=(FONT, 11), bg='#2b3a5c', fg=FG, activebackground='#3b4f7a',
                         activeforeground=FG, relief='flat', padx=10, pady=3, cursor='hand2', **kw)

    def _lbl(s, parent, text='', size=11, fg=FG, bold=False, bg=None, **kw):
        return tk.Label(parent, text=text, font=(FONT, size, 'bold' if bold else 'normal'), fg=fg, bg=bg or parent['bg'], **kw)

    def _build(s):
        r = s.root
        # 第 1 列：影像來源
        b1 = tk.Frame(r, bg=BG); b1.pack(fill='x', padx=12, pady=(10, 4))
        s._lbl(b1, '① 影像', 12, bold=True).pack(side='left', padx=(0, 8))
        s._lbl(b1, '車色').pack(side='left')
        s.cb = ttk.Combobox(b1, textvariable=s.color, width=16, state='readonly', font=(FONT, 11))
        s.cb.pack(side='left', padx=4); s.cb.bind('<<ComboboxSelected>>', lambda e: s._on_color())
        s._lbl(b1, '距離').pack(side='left', padx=(12, 2))
        s._btn(b1, '◀', lambda: s._step_t(-1)).pack(side='left')
        s.scale = ttk.Scale(b1, from_=1, to=12, orient='horizontal', length=260, command=lambda v: s._on_scale(v))
        s.scale.pack(side='left', padx=4)
        s._btn(b1, '▶', lambda: s._step_t(1)).pack(side='left')
        s.tlabel = s._lbl(b1, '', 12, fg='#f5b700', bold=True, width=14, anchor='w'); s.tlabel.pack(side='left', padx=6)
        s._lbl(b1, '張').pack(side='left')
        tk.Spinbox(b1, from_=1, to=10, textvariable=s.shot, width=3, font=(FONT, 11), command=s._load_from_browser).pack(side='left', padx=4)
        s._btn(b1, '開啟影像…', s.open_img).pack(side='left', padx=(16, 4))
        s._lbl(b1, '（也可把影像／遮罩檔拖進畫面）', 10, fg=MUTED).pack(side='left')
        # 第 2 列：遮罩與估測
        b2 = tk.Frame(r, bg=BG); b2.pack(fill='x', padx=12, pady=4)
        s._lbl(b2, '② 遮罩', 12, bold=True).pack(side='left', padx=(0, 8))
        for v, t in [('auto', '資料集遮罩（自動）'), ('sam', 'SAM 框選（拖一個框）'), ('paint', '手動塗抹')]:
            tk.Radiobutton(b2, text=t, value=v, variable=s.mode, command=s._on_mode, font=(FONT, 11), bg=BG, fg=FG,
                           selectcolor='#2b3a5c', activebackground=BG, activeforeground=FG).pack(side='left', padx=4)
        s._lbl(b2, '筆刷').pack(side='left', padx=(10, 2))
        ttk.Scale(b2, from_=5, to=120, variable=s.brush, orient='horizontal', length=100).pack(side='left')
        s._btn(b2, '指定遮罩檔…', s.open_mask).pack(side='left', padx=(10, 4))
        s._btn(b2, '清除遮罩', s.clear).pack(side='left', padx=4)
        s._lbl(b2, '③', 12, bold=True).pack(side='left', padx=(18, 4))
        tk.Checkbutton(b2, text='自動估測', variable=s.autorun, font=(FONT, 11), bg=BG, fg=FG, selectcolor='#2b3a5c',
                       activebackground=BG, activeforeground=FG).pack(side='left')
        tk.Button(b2, text='估測距離  (Enter)', command=s.run, font=(FONT, 12, 'bold'), bg='#f5b700', fg='#141b2d',
                  activebackground='#ffd24d', relief='flat', padx=14, cursor='hand2').pack(side='left', padx=8)
        # 主體：左影像、右結果
        body = tk.Frame(r, bg=BG); body.pack(fill='both', expand=True, padx=12, pady=6)
        body.columnconfigure(0, weight=1); body.rowconfigure(0, weight=1)
        s.cv = tk.Canvas(body, bg='black', highlightthickness=0, cursor='crosshair'); s.cv.grid(row=0, column=0, sticky='nsew')
        s.cv.bind('<Configure>', lambda e: s.show()); s.cv.bind('<ButtonPress-1>', s._press)
        s.cv.bind('<B1-Motion>', s._drag); s.cv.bind('<ButtonRelease-1>', s._release)
        s.cv.bind('<B3-Motion>', lambda e: s._paint(e, 0)); s.cv.bind('<Button-3>', lambda e: s._paint(e, 0))
        try:
            from tkinterdnd2 import DND_FILES
            s.cv.drop_target_register(DND_FILES); s.cv.dnd_bind('<<Drop>>', s._drop)
        except Exception:
            pass
        right = tk.Frame(body, bg=PANEL, width=640); right.grid(row=0, column=1, sticky='ns', padx=(12, 0)); right.grid_propagate(False)
        s._build_right(right)
        s.status = s._lbl(r, '', 10, fg=MUTED, anchor='w', bg='#0e1424'); s.status.pack(fill='x', side='bottom', ipady=3, ipadx=8)
        s._fill_colors(); s._hint()

    def _build_right(s, p):
        pad = dict(padx=14, sticky='ew')
        p.columnconfigure(0, weight=1)
        s.title = s._lbl(p, '請選擇影像', 13, bold=True, anchor='w', justify='left', wraplength=600); s.title.grid(row=0, pady=(12, 2), **pad)
        s.gtl = s._lbl(p, '', 22, fg='#ffffff', bold=True, anchor='w'); s.gtl.grid(row=1, **pad)
        s.maskl = s._lbl(p, '', 10, fg=MUTED, anchor='w', justify='left', wraplength=600); s.maskl.grid(row=2, pady=(0, 6), **pad)
        # 結果表
        tb = tk.Frame(p, bg=PANEL); tb.grid(row=3, **pad)
        heads = ['', '方法', '估測', '誤差', '時間', '硬體', '測試集RMSE']
        widths = [2, 17, 7, 7, 8, 5, 9]
        for j, (h, w) in enumerate(zip(heads, widths)):
            s._lbl(tb, h, 10, fg=MUTED, width=w, anchor='w' if j == 1 else 'e').grid(row=0, column=j, pady=(0, 2))
        s.cells = {}
        for i, (k, name, col, kind, note) in enumerate(METHODS, 1):
            row = [s._lbl(tb, '●', 13, fg=col, width=2)]
            bold = k == 'opt'
            row.append(s._lbl(tb, name, 11, fg=col, bold=bold, width=17, anchor='w'))
            for j in range(2, 7):
                row.append(s._lbl(tb, '', 11, fg=FG, bold=bold and j in (2, 3), width=widths[j], anchor='e'))
            for j, c in enumerate(row):
                c.grid(row=i, column=j, pady=1)
            r_ = RMSE.get(k); row[6].config(text=f'{r_:.3f} m' if r_ is not None else '—', fg=MUTED)
            row[5].config(text='GPU' if kind == 'deep' else 'CPU', fg=MUTED)
            s.cells[k] = row
        s._lbl(p, '誤差 = 估測 − 真實；★ = 本張最接近真值。測試集RMSE = 紅綠藍黑 4 色 × 1.2–6.7 m 共 480 張之整體 RMSE。',
               9, fg=MUTED, anchor='w', justify='left', wraplength=600).grid(row=4, pady=(4, 6), **pad)
        # 距離數線
        s._lbl(p, '距離數線（白線 = 真實距離）', 11, bold=True, anchor='w').grid(row=5, **pad)
        s.line = tk.Canvas(p, bg=PANEL, height=150, highlightthickness=0); s.line.grid(row=6, pady=(2, 8), **pad)
        s.line.bind('<Configure>', lambda e: s._draw_line())
        # 計時與硬體
        s.timer = s._lbl(p, '', 12, fg='#7ee787', bold=True, anchor='w'); s.timer.grid(row=7, **pad)
        s.hw = s._lbl(p, '', 10, fg=FG, anchor='w', justify='left', wraplength=600); s.hw.grid(row=8, pady=(4, 12), **pad)
        s.hw_cpu = f'CPU：{cpu_name()}（{os.cpu_count()} 執行緒）'; s.hw_gpu = 'GPU：偵測中…'; s._show_hw()

    def _hotkeys(s):
        r = s.root
        r.bind('<Return>', lambda e: s.run()); r.bind('<Left>', lambda e: s._step_t(-1)); r.bind('<Right>', lambda e: s._step_t(1))
        r.bind('<Up>', lambda e: s._step_shot(1)); r.bind('<Down>', lambda e: s._step_shot(-1))

    # ---- 執行緒安全的 UI 更新 ----
    def ui(s, fn):
        s.q.put(fn)

    def _poll(s):
        try:
            while True:
                s.q.get_nowait()()
        except queue.Empty:
            pass
        if s.t_start is not None:
            s.timer.config(text=f'⏱ 計算中… {time.perf_counter() - s.t_start:.1f} s')
        s.root.after(50, s._poll)

    def _quit(s):
        s.worker.close(); s.root.destroy()

    # ---- 硬體／狀態 ----
    def _worker_status(s, st_, msg):
        i = s.worker.info
        if i.get('gpu'):
            s.hw_gpu = f'GPU：{i["gpu"]}（{i["vram_gb"]} GB，CUDA {i["cuda"]}，sm_{i["sm"].replace(".", "")}）— {msg}'
        else:
            s.hw_gpu = f'GPU 深度模型：{msg}'
        s._show_hw()
        if st_ == 'ready' and s.img is not None and s.autorun.get() and s.mask is not None and s.mask.any():
            s.run()

    def _show_hw(s):
        s.hw.config(text=f'{s.hw_cpu}\n{s.hw_gpu}\nCA-DFDS／灰階：CPU（numpy + colour-science）；UniDepth、Depth Pro、SAM：GPU（PyTorch）')

    def _hint(s):
        h = {'auto': '遮罩：自動使用資料集手繪遮罩（各車色共用 firetruck_black 的遮罩）。找不到時請改用 SAM 框選。',
             'sam': 'SAM 框選：在影像上按住左鍵拖一個框把車框起來，放開後 SAM（GPU）自動分割出車輛範圍。',
             'paint': '手動塗抹：左鍵塗抹車輛範圍，右鍵擦除；「筆刷」調整大小。'}[s.mode.get()]
        s.status.config(text=h + '　｜　快捷鍵：←/→ 換距離、↑/↓ 換張數、Enter 估測')

    # ---- 資料集瀏覽 ----
    def _fill_colors(s):
        names = []
        for d in sorted(glob.glob(os.path.join(DATA, 'firetruck_*', DS))):
            names.append(os.path.basename(os.path.dirname(d)).replace('firetruck_', ''))
        for d in sorted(glob.glob(os.path.join(DATA, 'training', 'firetruck_*', DS))):
            names.append('校正-' + os.path.basename(os.path.dirname(d)).replace('firetruck_', ''))
        s.cb['values'] = names
        if names:
            s.color.set('red' if 'red' in names else names[0])
            s._on_color(load=False)

    def _ds_dir(s):
        c = s.color.get()
        if c.startswith('校正-'):
            return os.path.join(DATA, 'training', 'firetruck_' + c[3:], DS)
        return os.path.join(DATA, 'firetruck_' + c, DS)

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
        if not getattr(s, 'tdirs', None):
            return
        _, d = s._tdist(s.tidx.get())
        s.tlabel.config(text=f'T{s.tidx.get():02d}  {d:.1f} m' if d else f'T{s.tidx.get():02d}')

    def _on_scale(s, v):
        i = int(round(float(v)))
        if i != s.tidx.get():
            s.tidx.set(i); s._tlabel()
            if hasattr(s, '_deb'):
                s.root.after_cancel(s._deb)
            s._deb = s.root.after(250, s._load_from_browser)   # 拖曳時停下 0.25 秒才載入

    def _step_t(s, d):
        if getattr(s, 'tdirs', None):
            i = min(max(s.tidx.get() + d, 1), len(s.tdirs)); s.tidx.set(i); s.scale.set(i); s._tlabel(); s._load_from_browser()

    def _step_shot(s, d):
        s.shot.set(min(max(s.shot.get() + d, 1), 10)); s._load_from_browser()

    def _load_from_browser(s):
        if not getattr(s, 'tdirs', None):
            return
        files, _ = s._tdist(s.tidx.get())
        if files:
            s.load_image(files[min(s.shot.get(), len(files)) - 1])

    def _sync_browser(s, p):
        """影像從別處開啟時，讓上方車色／距離／張數同步顯示它在資料集中的位置。"""
        m = re.search(r'(training[\\/])?firetruck_([^\\/]+)[\\/][^\\/]+[\\/]T(\d+)[\\/]R[\\/]', p, re.I)
        if not m:
            return
        name = ('校正-' if m.group(1) else '') + m.group(2)
        if name in s.cb['values'] and name != s.color.get():
            s.color.set(name); s._on_color(load=False)
        s.tidx.set(int(m.group(3))); s.scale.set(int(m.group(3))); s._tlabel()
        files, _ = s._tdist(s.tidx.get()) if getattr(s, 'tdirs', None) else ([], None)
        norm = [os.path.normcase(os.path.abspath(f)) for f in files]
        if os.path.normcase(os.path.abspath(p)) in norm:
            s.shot.set(norm.index(os.path.normcase(os.path.abspath(p))) + 1)

    # ---- 載入 ----
    def open_img(s):
        init = os.path.dirname(s.path) if s.path else DATA
        p = filedialog.askopenfilename(initialdir=init, filetypes=[('Image', '*.png *.jpg *.jpeg *.bmp')])
        if p:
            s.load_image(p)

    def open_mask(s):
        if s.img is None:
            return
        _, d = auto_mask_path(s.path) if s.path else (None, DATA)
        p = filedialog.askopenfilename(initialdir=d if d and os.path.isdir(d) else DATA, filetypes=[('Mask', '*.npy *.png')])
        if p:
            s.set_mask(C.load_mask(p, s.img.shape), f'指定檔案 {os.path.basename(p)}'); s._maybe_run()

    def _drop(s, e):
        for p in s.root.tk.splitlist(e.data):
            if p.lower().endswith('.npy') or '_mask' in p.lower():
                if s.img is not None:
                    s.set_mask(C.load_mask(p, s.img.shape), f'拖入 {os.path.basename(p)}')
            else:
                s.load_image(p); return
        s._maybe_run()

    def load_image(s, p):
        img = C.imread(p)
        if img is None:
            messagebox.showerror('CA-DFDS', f'無法讀取影像：{p}'); return
        s.img, s.path = img, p; s.job += 1; s.res = {}; s.t_start = None; s.timer.config(text='')
        s._sync_browser(p)
        mp, _ = auto_mask_path(p)
        s.ref_mask = C.load_mask(mp, img.shape) if mp else None
        rel = os.path.relpath(p, DATA) if p.lower().startswith(DATA.lower()) else p
        s.title.config(text=rel)
        g = gt_of(p); s.gtl.config(text=f'真實距離 {g:.2f} m' if g else '真實距離：未知（檔名無 D<距離>_）')
        if s.mode.get() == 'auto' or s.mask is None or s.mask.shape != img.shape[:2]:
            if s.ref_mask is not None:
                s.set_mask(s.ref_mask.copy(), f'資料集手繪遮罩 {os.path.relpath(mp, DATA) if mp.lower().startswith(DATA.lower()) else mp}')
            else:
                s.set_mask(np.zeros(img.shape[:2], bool), '此影像沒有資料集遮罩 → 請用「SAM 框選」或「手動塗抹」')
        else:
            s.set_mask(s.mask, s.mask_src)
        s._clear_results(); s._maybe_run()

    def set_mask(s, m, src):
        s.mask = m.astype(bool); s.mask_src = src
        txt = f'遮罩：{src}'
        if m.any():
            txt += f'　面積 {100 * s.mask.mean():.1f}%'
            if s.ref_mask is not None and not np.array_equal(s.mask, s.ref_mask):
                iou = (s.mask & s.ref_mask).sum() / max((s.mask | s.ref_mask).sum(), 1)
                txt += f'　與資料集手繪遮罩 IoU = {iou:.2f}'
        s.maskl.config(text=txt); s.show()

    def clear(s):
        if s.img is not None:
            s.set_mask(np.zeros(s.img.shape[:2], bool), '已清除'); s._clear_results()

    def _on_mode(s):
        s._hint()
        if s.mode.get() == 'auto' and s.img is not None and s.ref_mask is not None:
            s.set_mask(s.ref_mask.copy(), '資料集遮罩'); s._maybe_run()

    # ---- 滑鼠 ----
    def _xy(s, e):
        return (e.x - s.ox) / s.sc, (e.y - s.oy) / s.sc

    def _press(s, e):
        if s.img is None:
            return
        if s.mode.get() == 'sam':
            s.box0 = s._xy(e)
        else:
            if s.mode.get() == 'auto':
                s.mode.set('paint'); s._hint()
            s._paint(e, 1)

    def _drag(s, e):
        if s.img is None:
            return
        if s.mode.get() == 'sam' and getattr(s, 'box0', None):
            s.cv.delete('rubber'); x0, y0 = s.box0
            s.cv.create_rectangle(x0 * s.sc + s.ox, y0 * s.sc + s.oy, e.x, e.y, outline='#38bdf8', width=2, dash=(6, 3), tags='rubber')
        elif s.mode.get() == 'paint':
            s._paint(e, 1)

    def _release(s, e):
        if s.img is None:
            return
        if s.mode.get() == 'sam' and getattr(s, 'box0', None):
            (x0, y0), (x1, y1) = s.box0, s._xy(e); s.box0 = None
            box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                return
            s._sam(box)
        elif s.mode.get() == 'paint':
            s._maybe_run()

    def _paint(s, e, v):
        if s.img is None:
            return
        x, y = s._xy(e); m = s.mask.astype(np.uint8)
        cv2.circle(m, (int(x), int(y)), int(s.brush.get() / s.sc), int(v), -1)
        s.mask = m > 0; s.mask_src = '手動塗抹'; s.show(fast=True)

    def _sam(s, box):
        if not s.worker.ready:
            messagebox.showinfo('CA-DFDS', 'GPU 模型尚未就緒（請看右下硬體狀態），或未安裝 torch 環境。'); return
        s.status.config(text='SAM 分割中…')
        img_path, out, job = s.path, os.path.join(TMP, 'sam.npy'), s.job

        def work():
            try:
                r = s.worker.call({'cmd': 'sam', 'image': img_path, 'point': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                                   'box': box, 'out': out})
                m = np.load(out) > 0
                s.ui(lambda: job == s.job and (s.set_mask(m, f'SAM 框選（GPU {r["ms"]:.0f} ms，分數 {r["score"]:.2f}）'),
                                               s._hint(), s._maybe_run()))
            except Exception as ex:
                s.ui(lambda: messagebox.showerror('SAM', str(ex)))
        threading.Thread(target=work, daemon=True).start()

    # ---- 繪圖 ----
    def show(s, fast=False):
        s.cv.delete('all')
        if s.img is None:
            w, h = s.cv.winfo_width(), s.cv.winfo_height()
            s.cv.create_text(w / 2, h / 2, text='用上方「車色／距離」選一張影像\n或按「開啟影像…」，或把影像檔拖進來',
                             fill=MUTED, font=(FONT, 16), justify='center'); return
        W, H = max(s.cv.winfo_width(), 100), max(s.cv.winfo_height(), 100)
        s.sc = min(W / s.img.shape[1], H / s.img.shape[0])
        nw, nh = int(s.img.shape[1] * s.sc), int(s.img.shape[0] * s.sc); s.ox, s.oy = (W - nw) // 2, (H - nh) // 2
        d = cv2.resize(s.img, (nw, nh), interpolation=cv2.INTER_NEAREST if fast else cv2.INTER_AREA)
        m = cv2.resize(s.mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        d[~m] = (0.55 * d[~m]).astype(np.uint8)          # 遮罩外調暗，車輛範圍保持原色
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(d, cs, -1, (94, 197, 34), 3)    # 綠色輪廓 = 遮罩邊界
        s.tkimg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(d, cv2.COLOR_BGR2RGB)))
        s.cv.create_image(s.ox, s.oy, anchor='nw', image=s.tkimg)
        if s.mask.any():
            ys, xs = np.nonzero(s.mask)
            x0, y0, x1, y1 = xs.min() * s.sc + s.ox, ys.min() * s.sc + s.oy, xs.max() * s.sc + s.ox, ys.max() * s.sc + s.oy
            s.cv.create_rectangle(x0, y0, x1, y1, outline='#f5b700', width=2)
            s._box_label(x0, y0, x1, y1)

    def _box_label(s, x0, y0, x1, y1):
        g = gt_of(s.path); lines = []
        for k, name, col, *_ in METHODS:
            if k in s.res and 'est' in s.res[k]:
                e = s.res[k]['est']; lines.append((f'{name}  {e:.2f} m' + (f'  ({e - g:+.2f})' if g else ''), col))
        if not lines:
            return
        ty = max(y0 - 6 - 22 * len(lines), 4) if y0 > 22 * len(lines) + 10 else y0 + 6
        bw = 330; bx = min(max(x0, 4), s.cv.winfo_width() - bw - 4)
        s.cv.create_rectangle(bx, ty - 2, bx + bw, ty + 22 * len(lines) + 2, fill='#0e1424', outline='#f5b700')
        for i, (t, col) in enumerate(lines):
            s.cv.create_text(bx + 8, ty + 11 + 22 * i, text=t, fill=col, anchor='w',
                             font=(FONT, 11, 'bold' if i == 0 else 'normal'))

    def _draw_line(s):
        c = s.line; c.delete('all'); W = c.winfo_width(); g = gt_of(s.path)
        lo, hi, L, R = 0.0, 8.0, 150, W - 20
        X = lambda v: L + (min(max(v, lo), hi) - lo) / (hi - lo) * (R - L)
        y_axis = 130
        c.create_line(L, y_axis, R, y_axis, fill=MUTED)
        for v in range(0, 9):
            c.create_line(X(v), y_axis - 4, X(v), y_axis + 4, fill=MUTED)
            c.create_text(X(v), y_axis + 12, text=f'{v} m', fill=MUTED, font=(FONT, 9))
        if g:
            c.create_line(X(g), 4, X(g), y_axis, fill='#ffffff', width=2, dash=(4, 2))
        for i, (k, name, col, *_) in enumerate(METHODS):
            y = 12 + i * 19
            c.create_text(L - 8, y, text=name, fill=col, anchor='e', font=(FONT, 9, 'bold' if k == 'opt' else 'normal'))
            r = s.res.get(k, {})
            if 'est' in r:
                e = r['est']; x = X(e)
                if g:
                    c.create_line(X(g), y, x, y, fill=col, width=3)
                c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=col, outline='#ffffff' if k == 'opt' else col, width=2)
                txt = f'{e:.2f}' + ('▶' if e > hi else '')
                c.create_text(x + (10 if x < R - 40 else -10), y, text=txt, fill=col, anchor='w' if x < R - 40 else 'e', font=(FONT, 9, 'bold'))

    # ---- 結果表 ----
    def _clear_results(s):
        s.res = {}
        for k, row in s.cells.items():
            row[0].config(text='●'); row[2].config(text=''); row[3].config(text='', fg=FG); row[4].config(text='')
        s._draw_line(); s.show()

    def _set_result(s, job, k, **r):
        if job != s.job:
            return
        s.res[k] = r; row = s.cells[k]; g = gt_of(s.path)
        if 'err' in r:
            row[2].config(text='失敗'); row[4].config(text=''); row[3].config(text='')
        else:
            row[2].config(text=f'{r["est"]:.2f} m'); row[4].config(text=f'{r["ms"]:.0f} ms')
            if g:
                e = r['est'] - g; row[3].config(text=f'{e:+.2f} m', fg='#7ee787' if abs(e) < 0.2 else '#ffb86b' if abs(e) < 0.5 else '#ff6b6b')
        if g:
            done = {kk: abs(v['est'] - g) for kk, v in s.res.items() if 'est' in v}
            best = min(done, key=done.get) if done else None
            for kk, rr in s.cells.items():
                rr[0].config(text='★' if kk == best else '●')
        s._draw_line(); s.show()

    # ---- 計算 ----
    def _maybe_run(s):
        if s.autorun.get() and s.img is not None and s.mask is not None and s.mask.sum() >= 100:
            s.run()

    def run(s):
        if s.img is None or s.mask is None or s.mask.sum() < 100:
            messagebox.showinfo('CA-DFDS', '請先選影像並標示車輛遮罩（資料集遮罩／SAM 框選／手動塗抹）'); return
        s.job += 1; job = s.job; s._clear_results(); s.t_start = time.perf_counter()
        img, mask, path = s.img, s.mask.copy(), s.path
        pending = {'cpu'}
        deep = [m[0] for m in METHODS if m[3] == 'deep'] if s.worker.ready else []
        if deep:
            pending.add('gpu')

        def finish(part):
            pending.discard(part)
            if not pending and job == s.job:
                tot = time.perf_counter() - s.t_start; s.t_start = None
                s.timer.config(text=f'⏱ 全部完成，總時間 {tot:.2f} s（CPU 與 GPU 平行計算）')

        def cpu():
            ci, cm = C.crop_to_mask(img, mask)
            t0 = time.perf_counter(); gs = C.gray_sigma(ci, cm); tg = 1000 * (time.perf_counter() - t0)
            if 'gray' in CAL:
                s.ui(lambda: s._set_result(job, 'gray', est=np.polyval(CAL['gray']['coef_cm'], gs) / 100, ms=tg, sigma=gs))
            t0 = time.perf_counter(); cov = C.laplacian_cov(ci, cm); tc = 1000 * (time.perf_counter() - t0)
            for k in ('opt', 'reopt5', 'jonly'):
                if k in CAL:
                    sg = C.sigma_eq5(cov, CAL[k]['weights'])
                    s.ui(lambda k=k, sg=sg: s._set_result(job, k, est=np.polyval(CAL[k]['coef_cm'], sg) / 100, ms=tc, sigma=sg))
            s.ui(lambda: finish('cpu'))

        def gpu():
            mp = os.path.join(TMP, f'mask_{job}.npy'); np.save(mp, mask.astype(np.uint8))
            for k in deep:
                if job != s.job:
                    return
                try:
                    r = s.worker.call({'cmd': 'depth', 'model': k, 'image': path, 'mask': mp})
                    s.ui(lambda k=k, r=r: s._set_result(job, k, est=r['median_m'], ms=r['ms']))
                except Exception as ex:
                    s.ui(lambda k=k, ex=ex: s._set_result(job, k, err=str(ex)))
            s.ui(lambda: finish('gpu'))
        threading.Thread(target=cpu, daemon=True).start()
        if deep:
            threading.Thread(target=gpu, daemon=True).start()


def grab_window(root):
    """Capture only this window's contents (Windows PrintWindow), unaffected by other windows or notifications on top, so they never leak into screenshots."""
    from PIL import Image as _I, ImageGrab
    try:
        import ctypes
        from ctypes import wintypes
        u, g = ctypes.windll.user32, ctypes.windll.gdi32
        hwnd = u.GetParent(root.winfo_id()) or root.winfo_id()
        r = wintypes.RECT(); u.GetClientRect(hwnd, ctypes.byref(r)); w, h = r.right, r.bottom
        hdc = u.GetDC(hwnd); mdc = g.CreateCompatibleDC(hdc); bmp = g.CreateCompatibleBitmap(hdc, w, h); g.SelectObject(mdc, bmp)
        u.PrintWindow(hwnd, mdc, 3)          # PW_CLIENTONLY | PW_RENDERFULLCONTENT
        class BIH(ctypes.Structure):
            _fields_ = [('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG), ('biPlanes', wintypes.WORD),
                        ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD), ('biSizeImage', wintypes.DWORD),
                        ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG), ('biClrUsed', wintypes.DWORD),
                        ('biClrImportant', wintypes.DWORD)]
        bi = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0); buf = ctypes.create_string_buffer(w * h * 4)
        g.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
        g.DeleteObject(bmp); g.DeleteDC(mdc); u.ReleaseDC(hwnd, hdc)
        return _I.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1).convert('RGB')
    except Exception:
        x, y = root.winfo_rootx(), root.winfo_rooty()
        return ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height()), all_screens=True)


def main():
    try:
        import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)   # 高 DPI 螢幕字體清晰、不被裁切
    except Exception:
        pass
    try:
        from tkinterdnd2 import TkinterDnD; root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    root.geometry('1800x1000'); root.minsize(1300, 760)
    try:
        root.state('zoomed')
    except Exception:
        pass
    app = App(root)
    if A.image:
        root.after(300, lambda: (app.load_image(A.image), A.mask and app.set_mask(C.load_mask(A.mask, app.img.shape), os.path.basename(A.mask))))
    if A.screenshot:
        def snap(n=0):
            busy = app.t_start is not None or (not app.worker.ready and n < 120 and not A.no_deep) or not app.res
            if busy and n < 240:
                root.after(500, lambda: snap(n + 1)); return
            root.update(); grab_window(root).save(A.screenshot)
            app._quit()
        root.after(1500, snap)
    root.mainloop()


if __name__ == '__main__':
    main()
