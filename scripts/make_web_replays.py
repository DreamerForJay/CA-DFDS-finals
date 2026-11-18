"""Render scenario replays from the original CARLA frames and recorded estimates."""
import csv
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CADFDS_DATA", ROOT.parent / "Carla-0.10.0-Win64-Shipping" / "datas"))
DEMO = ROOT / "results" / "demo"
OUT = DEMO / "scenario_replays"
FFMPEG = Path(os.environ.get("CADFDS_FFMPEG", ROOT / "logs" / "bin" / "ffmpeg.exe"))
DATASET = "Test_dataset_Focus1500cm_f1.4_f22.0"
SIZE = (896, 504)
HOLD = 16
BOXES = ROOT / "web_demo" / "figures" / "vehicle_boxes.json"


def render(name, rows):
    temp = OUT / f"{name}.tmp.mp4"
    cmd = [str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", "24", "-i", "-", "-c:v", "libx264",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        for index, row in enumerate(rows):
            photo = cv2.imread(str(OUT / f"{row['color']}_{row['gt_m']}m.jpg"))
            height = SIZE[1] - 69
            width = round(height * photo.shape[1] / photo.shape[0])
            frame = cv2.copyMakeBorder(cv2.resize(photo, (width, height)), 0, 69,
                                       (SIZE[0] - width) // 2, (SIZE[0] - width + 1) // 2,
                                       cv2.BORDER_CONSTANT, value=(23, 17, 10))
            caption = f"GT {row['gt_m']} m     CA-DFDS {float(row['cadfds_m']):.2f} m"
            cv2.putText(frame, caption, (20, 467), cv2.FONT_HERSHEY_SIMPLEX, .72, (205, 245, 122), 2, cv2.LINE_AA)
            detail = f"CARLA discrete samples | {row['color']} | {index + 1}/{len(rows)}"
            cv2.putText(frame, detail, (20, 491), cv2.FONT_HERSHEY_SIMPLEX, .51, (195, 189, 170), 1, cv2.LINE_AA)
            encoder.stdin.write(frame.tobytes() * HOLD)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError(f"FFmpeg failed for {name}")
    temp.replace(OUT / f"{name}.mp4")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["time_s", "color", "T", "gt_m", "cadfds_m"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"time_s": round(index * HOLD / 24, 3),
                             **{key: row[key] for key in writer.fieldnames[1:]}})


def main():
    OUT.mkdir(exist_ok=True)
    with (DEMO / "C-E1_method_compare" / "per_frame.csv").open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    boxes = {}
    for row in rows:
        color, distance = row["color"], row["gt_m"]
        step = int(row["T"])
        source = DATA / f"firetruck_{color}"
        path = source / DATASET / f"T{step:02d}" / "R" / f"D{distance}_shot01.png"
        photo = cv2.imread(str(path))
        if photo is None:
            raise FileNotFoundError(path)
        thumb = cv2.resize(photo, (720, 540), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(OUT / f"{color}_{distance}m.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 83]):
            raise OSError(f"Could not write {color} at {distance} m")
        mask_path = source / f"{DATASET}_mask" / f"T{step}.npy"
        mask = np.load(mask_path)
        ys, xs = np.nonzero(mask)
        if not xs.size:
            raise ValueError(f"Empty vehicle mask: {mask_path}")
        h, w = mask.shape
        box = (xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h)
        boxes[f"{color}_{distance}"] = [round(value, 4) for value in box]
    BOXES.parent.mkdir(parents=True, exist_ok=True)
    BOXES.write_text(json.dumps(boxes, indent=2) + "\n", encoding="utf-8")
    red = [row for row in rows if row["color"] == "red"]
    render("approach", red)
    render("roundtrip", red + red[-2::-1])
    render("colors", [row for row in rows if row["gt_m"] in {"1.2", "3.7", "6.7"}])


if __name__ == "__main__":
    main()
