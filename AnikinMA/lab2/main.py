import argparse
import os
import cv2

from detectors.models import YOLOv5DetectorONNX, YOLOv8DetectorONNX, YOLOXTinyDetectorONNX
from utils import list_images, read_annotations, match_detections_to_gt, compute_metrics, draw_predictions


def build_detector(model_name: str, models_dir: str, conf: float, nms: float):
    if model_name == "yolov5n_onnx":
        onnx = os.path.join(models_dir, "yolov5n_onnx", "yolov5n.onnx")
        names = os.path.join(models_dir, "yolov5n_onnx", "coco.names")
        return YOLOv5DetectorONNX(onnx, names, conf_threshold=conf, nms_threshold=nms)

    if model_name == "yolov8n_onnx":
        onnx = os.path.join(models_dir, "yolov8n_onnx", "yolov8n.onnx")
        names = os.path.join(models_dir, "yolov8n_onnx", "coco.names")
        return YOLOv8DetectorONNX(onnx, names, conf_threshold=conf, nms_threshold=nms)

    if model_name == "yolox_tiny_onnx":
        onnx = os.path.join(models_dir, "yolox_tiny_onnx", "yolox_tiny.onnx")
        names = os.path.join(models_dir, "yolox_tiny_onnx", "coco.names")
        return YOLOXTinyDetectorONNX(onnx, names, conf_threshold=conf, nms_threshold=nms)

    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", type=str, default="imgs_MOV03478")
    parser.add_argument("--ann", type=str, default="mov03478.txt")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--model", type=str, choices=["yolov5n_onnx", "yolov8n_onnx", "yolox_tiny_onnx"], default="yolov8n_onnx")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max_frames", type=int, default=-1)
    args = parser.parse_args()

    img_paths = list_images(args.imgs)
    gt_by_frame = read_annotations(args.ann)

    detector = build_detector(args.model, args.models_dir, args.conf, args.nms)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    n = len(img_paths) if args.max_frames < 0 else min(len(img_paths), args.max_frames)

    for frame_idx in range(n):
        img = cv2.imread(img_paths[frame_idx])
        if img is None:
            continue

        dets = detector.detect(img)
        gts = gt_by_frame.get(frame_idx, [])

        matches, tp, fp, fn = match_detections_to_gt(dets, gts, iou_threshold=args.iou)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if args.show:
            vis = img.copy()
            vis = draw_predictions(vis, dets, matches)
            tpr, fdr = compute_metrics(total_tp, total_fp, total_fn)
            cv2.putText(vis, f"TPR={tpr:.3f}  FDR={fdr:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 3)
            cv2.putText(vis, f"TPR={tpr:.3f}  FDR={fdr:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 1)
            cv2.imshow("Detections", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    tpr, fdr = compute_metrics(total_tp, total_fp, total_fn)
    print(f"Model: {args.model}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"TPR={tpr:.6f}")
    print(f"FDR={fdr:.6f}")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()