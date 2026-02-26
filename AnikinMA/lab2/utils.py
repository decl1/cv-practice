import os
import re
from typing import Dict, List, Tuple, Optional
import cv2

from detectors.base import Detection


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(img_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=natural_key)
    return [os.path.join(img_dir, f) for f in files]


def normalize_label(lbl: str) -> str:
    t = lbl.strip().lower()
    if t == "car":
        return "car"
    if t == "bus":
        return "bus"
    if t == "car ":
        return "car"
    if t == "bus ":
        return "bus"
    if t.upper() == "CAR":
        return "car"
    if t.upper() == "BUS":
        return "bus"
    return t


def read_annotations(path: str) -> Dict[int, List[Tuple[str, Tuple[int, int, int, int]]]]:
    gt: Dict[int, List[Tuple[str, Tuple[int, int, int, int]]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            label = normalize_label(parts[1])
            x1 = int(float(parts[2]))
            y1 = int(float(parts[3]))
            x2 = int(float(parts[4]))
            y2 = int(float(parts[5]))
            if label not in {"car", "bus"}:
                continue
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append((label, (x1, y1, x2, y2)))
    return gt


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def match_detections_to_gt(
    dets: List[Detection],
    gts: List[Tuple[str, Tuple[int, int, int, int]]],
    iou_threshold: float = 0.5,
):
    gts = [(normalize_label(lbl), box) for (lbl, box) in gts]
    gt_used = [False] * len(gts)
    matches: Dict[int, Optional[str]] = {}

    tp = 0
    fp = 0

    dets_sorted = sorted(dets, key=lambda x: float(x.confidence), reverse=True)

    for d in dets_sorted:
        d_lbl = normalize_label(d.label)
        if d_lbl not in {"car", "bus"}:
            continue

        best_j = -1
        best_iou = 0.0
        for gj, (g_lbl, g_box) in enumerate(gts):
            if gt_used[gj]:
                continue
            if g_lbl != d_lbl:
                continue
            v = iou(d.box, g_box)
            if v > best_iou:
                best_iou = v
                best_j = gj

        if best_j >= 0 and best_iou >= iou_threshold:
            gt_used[best_j] = True
            tp += 1
            matches[id(d)] = gts[best_j][0]
        else:
            fp += 1
            matches[id(d)] = None

    fn = int(sum(1 for u in gt_used if not u))
    return matches, tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int):
    denom_tpr = tp + fn
    denom_fdr = tp + fp
    tpr = 0.0 if denom_tpr == 0 else tp / denom_tpr
    fdr = 0.0 if denom_fdr == 0 else fp / denom_fdr
    return float(tpr), float(fdr)


def color_for_label(label: str):
    if label == "car":
        return (0, 200, 255)
    if label == "bus":
        return (0, 255, 0)
    return (200, 200, 200)


def draw_predictions(image, dets: List[Detection], matches: Dict[int, Optional[str]]):
    for d in dets:
        if normalize_label(d.label) not in {"car", "bus"}:
            continue
        x1, y1, x2, y2 = d.box
        c = color_for_label(normalize_label(d.label))
        cv2.rectangle(image, (x1, y1), (x2, y2), c, 2)

        label_text = f"{normalize_label(d.label)} {float(d.confidence):.3f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 + th + 2)
        cv2.putText(image, label_text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3)
        cv2.putText(image, label_text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1)

        gt = matches.get(id(d), None)
        gt_text = "NONE" if gt is None else gt
        gy = max(0, y1 - 6)
        cv2.putText(image, gt_text, (x1, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3)
        cv2.putText(image, gt_text, (x1, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return image