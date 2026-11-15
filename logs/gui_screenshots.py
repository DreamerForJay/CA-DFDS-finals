# -*- coding: utf-8 -*-
"""Take automatic GUI screenshots (2026-11-16 v2: calls gui_new_v3.py --screenshot directly; the window is captured with PrintWindow, so other windows cannot leak in).

Usage: python logs/gui_screenshots.py <output folder> <image1> [<image2> ...]
  The mask is loaded automatically from the dataset; output files are named gui_new_v3_1.png, gui_new_v3_2.png, ...
  The first run waits for the GPU models to load (about 20–40 s).
"""
import os, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, imgs = sys.argv[1], sys.argv[2:]
for i, p in enumerate(imgs, 1):
    dst = os.path.join(out, f'gui_new_v3_{i}.png')
    subprocess.run([sys.executable, os.path.join(ROOT, 'gui_new_v3.py'), '--image', p, '--screenshot', dst], check=True, cwd=ROOT)
    print('saved', dst)
