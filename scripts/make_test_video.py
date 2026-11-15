# -*- coding: utf-8 -*-
"""Build a test video "approaching a car ahead" from the dataset images (for the GUI video mode only; CARLA has no continuous footage).
Each distance T12→T01 (6.7→1.2 m) contributes shot01–03, each held for 10 frames at 10 fps. Every distance step is a jump (not continuous motion).
Usage: python scripts/make_test_video.py [color] [output]   default red → logs/scratch/approach_red.mp4
"""
import os, sys, glob, cv2
DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
c = sys.argv[1] if len(sys.argv) > 1 else 'red'
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'scratch', f'approach_{c}.mp4')
os.makedirs(os.path.dirname(out), exist_ok=True)
ds = os.path.join(DATA, f'firetruck_{c}', 'Test_dataset_Focus1500cm_f1.4_f22.0')
vw = None
for t in range(12, 0, -1):
    for f in sorted(glob.glob(os.path.join(ds, f'T{t:02d}', 'R', '*.png')))[:3]:
        img = cv2.imread(f)
        if vw is None:
            vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 10, (img.shape[1], img.shape[0]))
        for _ in range(10):
            vw.write(img)
vw.release(); print('saved', out)
