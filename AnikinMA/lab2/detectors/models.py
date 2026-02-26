from typing import List, Tuple
import cv2
import numpy as np

from .base import BaseDetector, Detection


def _load_names(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def _normalize_label(x: str) -> str:
    t = x.strip().lower()
    if t == "car":
        return "car"
    if t == "bus":
        return "bus"
    return t


def _nms_xyxy(boxes: List[Tuple[int, int, int, int]], scores: List[float], conf_thr: float, nms_thr: float):
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in boxes],
        scores=[float(s) for s in scores],
        score_threshold=float(conf_thr),
        nms_threshold=float(nms_thr),
    )
    if idxs is None or len(idxs) == 0:
        return []
    return idxs.flatten().tolist()


def _letterbox(image, new_shape: Tuple[int, int], color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_h, new_w = new_shape
    r = min(new_w / w, new_h / h)
    resized_w = int(round(w * r))
    resized_h = int(round(h * r))
    img = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def _to_numpy(x):
    return np.array(x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class YOLOv5DetectorONNX(BaseDetector):
    def __init__(self, onnx_path: str, names_path: str, conf_threshold: float = 0.25, nms_threshold: float = 0.45, input_size: int = 640):
        super().__init__(conf_threshold, nms_threshold)
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.names = _load_names(names_path)
        self.input_size = int(input_size)
        self.keep = {"car", "bus"}

    def detect(self, image) -> List[Detection]:
        h, w = image.shape[:2]
        img, r, pad = _letterbox(image, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = np.array(out)

        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]

        if out.ndim == 2 and out.shape[0] in (84, 85) and out.shape[1] > out.shape[0]:
            out = out.T

        if out.ndim != 2 or out.shape[1] < 6:
            return []

        boxes = out[:, 0:4].astype(np.float32)
        cls = out[:, 4:].astype(np.float32)

        cls_max = float(np.max(cls)) if cls.size else 0.0
        if cls_max > 1.5:
            cls = _sigmoid(cls)

        cls_id = np.argmax(cls, axis=1)
        conf = cls[np.arange(cls.shape[0]), cls_id]

        keep = conf >= float(self.conf_threshold)
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]

        x = boxes[:, 0]
        y = boxes[:, 1]
        bw = boxes[:, 2]
        bh = boxes[:, 3]

        x1 = x - bw / 2.0
        y1 = y - bh / 2.0
        x2 = x + bw / 2.0
        y2 = y + bh / 2.0

        x1 = (x1 - pad[0]) / r
        y1 = (y1 - pad[1]) / r
        x2 = (x2 - pad[0]) / r
        y2 = (y2 - pad[1]) / r

        x1 = np.clip(x1, 0, w - 1).astype(np.int32)
        y1 = np.clip(y1, 0, h - 1).astype(np.int32)
        x2 = np.clip(x2, 0, w - 1).astype(np.int32)
        y2 = np.clip(y2, 0, h - 1).astype(np.int32)

        boxes_xyxy = []
        scores = []
        labels = []

        for i in range(len(conf)):
            cid = int(cls_id[i])
            name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            lab = _normalize_label(name)
            if lab not in self.keep:
                continue
            if x2[i] <= x1[i] or y2[i] <= y1[i]:
                continue
            boxes_xyxy.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])))
            scores.append(float(conf[i]))
            labels.append(lab)

        idxs = _nms_xyxy(boxes_xyxy, scores, self.conf_threshold, self.nms_threshold)
        dets: List[Detection] = []
        for i in idxs:
            dets.append(Detection(label=labels[i], confidence=float(scores[i]), box=boxes_xyxy[i]))
        return dets


class YOLOv8DetectorONNX(BaseDetector):
    def __init__(self, onnx_path: str, names_path: str, conf_threshold: float = 0.25, nms_threshold: float = 0.45, input_size: int = 640):
        super().__init__(conf_threshold, nms_threshold)
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.names = _load_names(names_path)
        self.input_size = int(input_size)
        self.keep = {"car", "bus"}

    def detect(self, image) -> List[Detection]:
        h, w = image.shape[:2]
        img, r, pad = _letterbox(image, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = np.array(out)

        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]

        if out.ndim == 2 and out.shape[0] in (84, 85) and out.shape[1] > out.shape[0]:
            out = out.T
        elif out.ndim == 3:
            out = out.reshape(-1, out.shape[-1])

        if out.ndim != 2 or out.shape[1] < 6:
            return []

        boxes = out[:, 0:4].astype(np.float32)
        cls = out[:, 4:].astype(np.float32)

        cls_max = float(np.max(cls)) if cls.size else 0.0
        if cls_max > 1.5:
            cls = _sigmoid(cls)

        cls_id = np.argmax(cls, axis=1)
        conf = cls[np.arange(cls.shape[0]), cls_id]

        keep = conf >= float(self.conf_threshold)
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]

        x = boxes[:, 0]
        y = boxes[:, 1]
        bw = boxes[:, 2]
        bh = boxes[:, 3]

        x1 = x - bw / 2.0
        y1 = y - bh / 2.0
        x2 = x + bw / 2.0
        y2 = y + bh / 2.0

        x1 = (x1 - pad[0]) / r
        y1 = (y1 - pad[1]) / r
        x2 = (x2 - pad[0]) / r
        y2 = (y2 - pad[1]) / r

        x1 = np.clip(x1, 0, w - 1).astype(np.int32)
        y1 = np.clip(y1, 0, h - 1).astype(np.int32)
        x2 = np.clip(x2, 0, w - 1).astype(np.int32)
        y2 = np.clip(y2, 0, h - 1).astype(np.int32)

        boxes_xyxy = []
        scores = []
        labels = []

        for i in range(len(conf)):
            cid = int(cls_id[i])
            name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            lab = _normalize_label(name)
            if lab not in self.keep:
                continue
            if x2[i] <= x1[i] or y2[i] <= y1[i]:
                continue
            boxes_xyxy.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])))
            scores.append(float(conf[i]))
            labels.append(lab)

        idxs = _nms_xyxy(boxes_xyxy, scores, self.conf_threshold, self.nms_threshold)
        dets: List[Detection] = []
        for i in idxs:
            dets.append(Detection(label=labels[i], confidence=float(scores[i]), box=boxes_xyxy[i]))
        return dets


class YOLOXTinyDetectorONNX(BaseDetector):
    def __init__(self, onnx_path: str, names_path: str, conf_threshold: float = 0.25, nms_threshold: float = 0.45, input_size: int = 640):
        super().__init__(conf_threshold, nms_threshold)
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.names = _load_names(names_path)
        self.input_size = int(input_size)
        self.strides = [8, 16, 32]
        self.keep = {"car", "bus"}

    def detect(self, image) -> List[Detection]:
        h, w = image.shape[:2]
        img, r, pad = _letterbox(image, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = np.squeeze(out, axis=0)

        boxes, scores, cls_ids = self._decode(out, self.input_size, self.input_size)
        if boxes.size == 0:
            return []

        x1 = (boxes[:, 0] - pad[0]) / r
        y1 = (boxes[:, 1] - pad[1]) / r
        x2 = (boxes[:, 2] - pad[0]) / r
        y2 = (boxes[:, 3] - pad[1]) / r

        x1 = np.clip(x1, 0, w - 1).astype(np.int32)
        y1 = np.clip(y1, 0, h - 1).astype(np.int32)
        x2 = np.clip(x2, 0, w - 1).astype(np.int32)
        y2 = np.clip(y2, 0, h - 1).astype(np.int32)

        boxes_xyxy = []
        scr = []
        lbls = []

        for i in range(len(scores)):
            if x2[i] <= x1[i] or y2[i] <= y1[i]:
                continue
            cid = int(cls_ids[i])
            name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            lab = _normalize_label(name)
            if lab not in self.keep:
                continue
            boxes_xyxy.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])))
            scr.append(float(scores[i]))
            lbls.append(lab)

        idxs = _nms_xyxy(boxes_xyxy, scr, self.conf_threshold, self.nms_threshold)
        dets: List[Detection] = []
        for i in idxs:
            dets.append(Detection(label=lbls[i], confidence=float(scr[i]), box=boxes_xyxy[i]))
        return dets

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _decode(self, preds: np.ndarray, in_w: int, in_h: int):
        num_classes = preds.shape[1] - 5

        grids = []
        expanded_strides = []
        for stride in self.strides:
            hs = in_h // stride
            ws = in_w // stride
            yv, xv = np.meshgrid(np.arange(hs), np.arange(ws), indexing="ij")
            grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))

        grid = np.concatenate(grids, axis=0).astype(np.float32)
        stride = np.concatenate(expanded_strides, axis=0).astype(np.float32)

        xys = (preds[:, 0:2] + grid) * stride
        whs = np.exp(preds[:, 2:4]) * stride
        x1y1 = xys - whs / 2.0
        x2y2 = xys + whs / 2.0
        boxes = np.concatenate([x1y1, x2y2], axis=1)

        obj = self._sigmoid(preds[:, 4:5])
        cls = self._sigmoid(preds[:, 5:5 + num_classes])
        conf_all = obj * cls
        conf = conf_all.max(axis=1)
        cls_id = conf_all.argmax(axis=1)

        keep = conf >= self.conf_threshold
        boxes = boxes[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]

        return boxes.astype(np.float32), conf.astype(np.float32), cls_id.astype(np.int32)