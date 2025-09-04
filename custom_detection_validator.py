#!/usr/bin/env python3
"""
Validate YOLO with a custom validator that:
- Uses a custom IoU range for mAP (map_min:map_max:0.05)
- Visualizes K images with TP/FP/FN overlays at IoU = map_min
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import box_iou


class CustomDetectionValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        # ---- Pop our custom args (so ultralytics get_cfg() doesn't error) ----
        self.map_min = float(args.pop("map_min", 0.5))   # inclusive
        self.map_max = float(args.pop("map_max", 0.95))  # inclusive-ish
        self.viz_k   = int(args.pop("viz_k", 50))
        super().__init__(dataloader, save_dir, args, _callbacks)

        # Replace the IoU thresholds used by YOLO’s matching/mAP
        steps = int(round((self.map_max - self.map_min) / 0.05)) + 1
        self.iouv = torch.linspace(self.map_min, self.map_min + 0.05 * (steps - 1), steps)
        self.niou = self.iouv.numel()

        # visualization setup
        self._viz_outdir: Optional[Path] = None
        self._viz_count: int = 0
        if self.viz_k != 0:
            try:
                self._viz_outdir = Path(self.save_dir) / f"visualizations_minIoU_{int(round(self.map_min * 100))}"
                self._viz_outdir.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Will save visualizations to: {self._viz_outdir}")
            except Exception as e:
                LOGGER.warning(f"Visualization dir create failed: {e}")
                self._viz_outdir = None

    # Show your chosen minimum IoU instead of "mAP50" in the table header
    def get_desc(self) -> str:
        lab = int(round(self.map_min * 100))
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", f"mAP{lab}", f"mAP{lab}-95)")

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        return super()._prepare_batch(si, batch)

    def _prepare_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super()._prepare_pred(pred)

    # ---------- drawing helpers ----------
    def _scale_to_original(self, boxes_xyxy_net: torch.Tensor, pbatch: Dict[str, Any]) -> np.ndarray:
        """Map net-input-space xyxy back to original image space."""
        if boxes_xyxy_net.numel() == 0:
            return np.zeros((0, 4), dtype=np.float32)
        b = boxes_xyxy_net.clone()
        b = ops.scale_boxes(
            pbatch["imgsz"], b, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        ).cpu().numpy()
        return b

    def _draw_boxes(self, im: np.ndarray, boxes: np.ndarray, color: Tuple[int, int, int],
                    labels: Optional[List[str]] = None):
        for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            if labels and i < len(labels):
                cv2.putText(im, labels[i], (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _match_at_threshold(self, gt_xyxy: torch.Tensor, gt_cls: torch.Tensor,
                            pred_xyxy: torch.Tensor, pred_cls: torch.Tensor,
                            thr: float) -> Dict[str, Any]:
        """
        Greedy matching at IoU=thr with class check.
        Returns dict with tp_mask (bool per pred), fp_idx (unmatched preds), fn_idx (unmatched gts).
        """
        ng, npred = gt_xyxy.shape[0], pred_xyxy.shape[0]
        if ng == 0 and npred == 0:
            return {"tp_mask": np.zeros((0,), bool), "fp_idx": np.array([], int), "fn_idx": np.array([], int)}
        if ng == 0:
            return {"tp_mask": np.zeros((npred,), bool), "fp_idx": np.arange(npred, dtype=int), "fn_idx": np.array([], int)}
        if npred == 0:
            return {"tp_mask": np.zeros((0,), bool), "fp_idx": np.array([], int), "fn_idx": np.arange(ng, dtype=int)}

        iou = box_iou(gt_xyxy, pred_xyxy).cpu().numpy()  # (ng, np)
        class_ok = (gt_cls.view(-1, 1).cpu().numpy() == pred_cls.view(1, -1).cpu().numpy())
        iou = np.where(class_ok, iou, 0.0)

        matches = np.argwhere(iou >= thr)
        if matches.shape[0]:
            # sort by IoU desc
            matches = matches[np.argsort(iou[matches[:, 0], matches[:, 1]])[::-1]]
            # unique by pred (keep best GT)
            _, keep_pred = np.unique(matches[:, 1], return_index=True)
            matches = matches[keep_pred]
            # unique by GT (keep best pred)
            _, keep_gt = np.unique(matches[:, 0], return_index=True)
            matches = matches[keep_gt]

        matched_gt = set(matches[:, 0].tolist()) if matches.size else set()
        matched_pred = set(matches[:, 1].tolist()) if matches.size else set()

        tp_mask = np.zeros((npred,), dtype=bool)
        if matches.size:
            tp_mask[list(matched_pred)] = True

        fp_idx = np.array([j for j in range(npred) if j not in matched_pred], dtype=int)
        fn_idx = np.array([i for i in range(ng) if i not in matched_gt], dtype=int)
        return {"tp_mask": tp_mask, "fp_idx": fp_idx, "fn_idx": fn_idx}

    # ---------- main metric update with on-the-fly viz ----------
    def update_metrics(self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Same as YOLO’s, but:
          - pass iou_thres=self.map_min to confusion_matrix so FP/TP/FN follow your min IoU
          - save first K visualizations with TP/FP/FN overlays
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            # confusion matrix at IoU = map_min (not YOLO's default)
            if self.args.plots:
                self.confusion_matrix.process_batch(
                    predn, pbatch, conf=self.args.conf, iou_thres=self.map_min
                )

            # ---- Visualize first K images (-1 means all, 0 means none) ----
            do_viz_all = (self.viz_k == -1)
            if self._viz_outdir is not None and (do_viz_all or self._viz_count < self.viz_k):
                # Compute TP/FP/FN at map_min in NET space
                match = self._match_at_threshold(
                    gt_xyxy=pbatch["bboxes"],  # already xyxy * img_size in _prepare_batch
                    gt_cls=pbatch["cls"],
                    pred_xyxy=predn["bboxes"],
                    pred_cls=predn["cls"],
                    thr=float(self.map_min),
                )

                # Scale boxes back to original image space for drawing
                gt_xyxy_orig   = self._scale_to_original(pbatch["bboxes"], pbatch)
                pred_xyxy_orig = self._scale_to_original(predn["bboxes"], pbatch)

                tp_mask = match["tp_mask"]
                fp_idx  = match["fp_idx"]
                fn_idx  = match["fn_idx"]

                tp_boxes = pred_xyxy_orig[tp_mask] if pred_xyxy_orig.size else np.zeros((0, 4))
                fp_boxes = pred_xyxy_orig[fp_idx]  if pred_xyxy_orig.size and len(fp_idx) else np.zeros((0, 4))
                fn_boxes = gt_xyxy_orig[fn_idx]    if gt_xyxy_orig.size and len(fn_idx) else np.zeros((0, 4))

                # Labels for predictions (class and conf)
                names = getattr(self, "names", None) or {}
                pred_cls_np  = predn["cls"].cpu().numpy().astype(int)
                pred_conf_np = predn["conf"].cpu().numpy()
                pred_labels  = [f"{names.get(int(c), int(c))}:{conf:.2f}" for c, conf in zip(pred_cls_np, pred_conf_np)]
                tp_labels = [pred_labels[i] for i, m in enumerate(tp_mask) if m]
                fp_labels = [pred_labels[i] for i in fp_idx] if len(fp_idx) else []

                # Read image
                im_path = pbatch["im_file"]
                im = cv2.imread(im_path)
                if im is None:
                    LOGGER.warning(f"[viz] Could not read image: {im_path}")
                else:
                    lab = int(round(self.map_min * 100))
                    # Draw GT first (magenta), then TP (green), FP (red), FN (yellow)
                    self._draw_boxes(im, gt_xyxy_orig, (255,   0, 255), labels=["GT"] * len(gt_xyxy_orig))
                    self._draw_boxes(im, tp_boxes,    (  0, 255,   0), labels=tp_labels)
                    self._draw_boxes(im, fp_boxes,    (  0,   0, 255), labels=fp_labels)
                    self._draw_boxes(im, fn_boxes,    (  0, 255, 255), labels=["FN"] * len(fn_boxes))

                    stem = Path(im_path).stem
                    out = self._viz_outdir / f"{stem}_viz_iou{lab}.jpg"
                    cv2.imwrite(str(out), im)
                    self._viz_count += 1

            if no_pred:
                continue

            # Save COCO JSON / TXT as usual
            if self.args.save_json:
                self.pred_to_json(predn, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        super().finalize_metrics()
        # nothing extra to write here now (we saved overlays during update)


def validate_yolov8(
    model_cfg: str,
    data_cfg: str,
    project: str,
    name: str,
    imgsz: Tuple[int, int] | int = (1088, 1440),
    batch: int = 1,
    workers: int = 1,
    conf: float = 0.1,
    iou: float = 0.3,
    max_det: int = 300,
    agnostic_nms: bool = False,
    plots: bool = True,
    save_json: bool = False,
    verbose: bool = True,
    *,
    map_min: float = 0.5,
    map_max: float = 0.95,
    viz_k: int = 50,         # 0: none, -1: all, >0: first K
):
    model = YOLO(model_cfg)
    results = model.val(
        data=data_cfg,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        conf=conf,
        iou=iou,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        plots=plots,
        save_json=save_json,
        verbose=verbose,
        name=name,
        project=project,
        rect=True,
        validator=CustomDetectionValidator,
        # custom knobs:
        map_min=map_min,
        map_max=map_max,
        viz_k=viz_k,
        visualize=True,    # needed so ConfusionMatrix stores matches (and enables on_val plotting)
    )
    LOGGER.info(f"Done. Results saved to: {Path(results.save_dir)}")
    return results


if __name__ == "__main__":
    pwd = os.path.dirname(os.path.abspath(__file__))

    MODEL = os.path.join(pwd, "runs", "train", "yolov8_large_rivendale_v5_k_v2_fold5", "weights", "best.pt")
    DATASET = "rivendale_v5_k_fold_v2"
    DATA_CFG = os.path.join(pwd, "datasets", DATASET, "data.yaml")
    OUTPUT_DIR = os.path.join(pwd, "runs", "detect")
    NAME = "yolov8_large_rivendale_v5_k_v2_fold5_custom_validator"

    # match your path setup
    os.chdir(os.path.join(pwd, "datasets", DATASET))
    print("CWD:", os.getcwd())

    validate_yolov8(
        model_cfg=MODEL,
        data_cfg=DATA_CFG,
        project=OUTPUT_DIR,
        name=NAME,
        imgsz=(1088, 1440),
        batch=1,
        workers=1,
        conf=0.1,
        iou=0.1,
        max_det=300,
        agnostic_nms=False,
        plots=True,
        save_json=False,
        verbose=True,
        map_min=0.30,    # e.g., 0.30 if you want to start mAP at 0.30
        map_max=0.95,
        viz_k=-1,        # 0: none, -1: all, >0: first K
    )
