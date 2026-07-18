"""Host-side decode for Hailo yolov8-seg hefs (pure numpy + cv2).

The Model Zoo yolov8_seg hefs have NO on-chip postprocess: the device
returns 10 raw tensors (3 scales x [box-distribution, class-scores,
mask-coefficients] + mask prototypes). This module turns them into
per-instance boolean masks + boxes + classes.

Adapted from hailo-apps (MIT) standalone instance_segmentation
post_process/postprocessing.py, with its two native-code dependencies
replaced so the Pi needs nothing beyond numpy/cv2:
- scipy.special.expit  -> local sigmoid
- cython_nms           -> vectorized numpy NMS

Config constants match hailo-apps config.json for the v8 arch
(strides 8/16/32, reg_max 15, 32 mask channels, 640x640 input).
"""

from __future__ import annotations

from typing import Any

import numpy as np


V8_STRIDES = (8, 16, 32)
V8_REG_MAX = 15
V8_NUM_CLASSES = 80
V8_INPUT_SHAPE = (640, 640)
V8_SCORE_THRESHOLD = 0.001
V8_NMS_IOU = 0.7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Classic greedy NMS; boxes (N,4) xyxy, scores (N,). Returns kept idx
    in descending-score order (inputs are pre-sorted by the caller)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.arange(len(boxes))  # caller sorts by confidence already
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_thres]
    return np.asarray(keep, dtype=int)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    nm: int = 32,
    multi_label: bool = True,
) -> list[dict[str, np.ndarray]]:
    """prediction: (batch, proposals, 4+1+classes+nm) — xywh, objectness,
    class scores, mask coefficients."""
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    mi = 5 + nc
    output: list[dict[str, np.ndarray]] = []

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4)),
                    "mask": np.zeros((0, nm)),
                    "detection_classes": np.zeros((0,)),
                    "detection_scores": np.zeros((0,)),
                }
            )
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = objectness * class score
        boxes = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        use_multi = multi_label and nc > 1
        if not use_multi:
            conf = np.expand_dims(x[:, 5:mi].max(1), 1)
            j = np.expand_dims(x[:, 5:mi].argmax(1), 1).astype(np.float32)
            keep_rows = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, mask), 1)[keep_rows]
        else:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate(
                (boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1
            )

        x = x[x[:, 4].argsort()[::-1]]

        # Per-class NMS via the class-offset trick.
        cls_shift = x[:, 5:6] * max_wh
        keep = numpy_nms(x[:, :4] + cls_shift, x[:, 4], iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]
        out = x[keep]

        output.append(
            {
                "detection_boxes": out[:, :4],
                "mask": out[:, 6:],
                "detection_classes": out[:, 5],
                "detection_scores": out[:, 4],
            }
        )
    return output


def crop_mask_roi_vectorized(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Zero out mask pixels outside each detection's box. masks (N,H,W),
    boxes (N,4) in mask pixels."""
    n, h, w = masks.shape
    output = np.zeros_like(masks)
    boxes = np.round(boxes).astype(int)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        output[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return output


def fast_resize_masks(masks: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    import cv2

    ih, iw = out_shape
    resized = np.empty((masks.shape[0], ih, iw), dtype=np.float32)
    for i in range(masks.shape[0]):
        resized[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)
    return resized


def process_mask(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """protos (mh,mw,32), masks_in (N,32), bboxes (N,4) in input pixels ->
    (N, ih, iw) float masks cropped to boxes."""
    mh, mw, c = protos.shape
    ih, iw = shape
    if masks_in.shape[0] == 0:
        return np.zeros((0, ih, iw), dtype=np.float32)
    protos_flat = protos.reshape(-1, c).T  # (32, mh*mw)
    masks = _sigmoid(masks_in.astype(np.float32) @ protos_flat).reshape(-1, mh, mw)
    masks = fast_resize_masks(masks, (ih, iw))
    return crop_mask_roi_vectorized(masks, bboxes)


def _yolov8_decoding(
    raw_boxes: list[np.ndarray],
    strides: tuple[int, ...],
    image_dims: tuple[int, int],
    reg_max: int,
) -> np.ndarray:
    boxes = None
    for box_distribute, stride in zip(raw_boxes, strides):
        shape = [int(x / stride) for x in image_dims]
        grid_x = np.arange(shape[1]) + 0.5
        grid_y = np.arange(shape[0]) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.flatten() * stride
        ct_col = grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

        reg_range = np.arange(reg_max + 1)
        box_distribute = np.reshape(
            box_distribute,
            (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1),
        )
        box_distance = _softmax(box_distribute)
        box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
        box_distance = np.sum(box_distance, axis=-1) * stride
        box_distance = np.concatenate(
            [box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1
        )
        decode_box = np.expand_dims(center, axis=0) + box_distance

        xmin, ymin = decode_box[:, :, 0], decode_box[:, :, 1]
        xmax, ymax = decode_box[:, :, 2], decode_box[:, :, 3]
        xywh_box = np.transpose(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0]
        )
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
    return boxes


def order_endnodes(outputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Arrange the hef's 10 raw output tensors into the fixed order the
    decoder expects: per scale 20->40->80 the (boxes, scores, coeffs)
    triple, then the 160x160 prototypes. Identified by shape, so output
    layer names don't matter."""
    protos = None
    by_scale: dict[int, dict[str, np.ndarray]] = {}
    for name, arr in outputs.items():
        if arr.ndim == 3:  # (H, W, C) — add batch dim
            arr = arr[np.newaxis, ...]
        _, h, w, c = arr.shape
        if h == 160:
            protos = arr
            continue
        role = {4 * (V8_REG_MAX + 1): "boxes", V8_NUM_CLASSES: "scores", 32: "coeffs"}.get(c)
        if role is None:
            raise ValueError(f"Unexpected seg output {name} with shape {arr.shape}")
        by_scale.setdefault(h, {})[role] = arr
    if protos is None or sorted(by_scale) != [20, 40, 80]:
        raise ValueError(f"Unexpected seg output set: scales={sorted(by_scale)}, protos={protos is not None}")
    endnodes = []
    for scale in (20, 40, 80):
        triple = by_scale[scale]
        endnodes += [triple["boxes"], triple["scores"], triple["coeffs"]]
    endnodes.append(protos)
    return endnodes


def yolov8_seg_postprocess(
    outputs: dict[str, np.ndarray],
    score_threshold: float = 0.25,
    nms_iou: float = V8_NMS_IOU,
    max_det: int = 50,
) -> dict[str, Any]:
    """Full decode: raw hef outputs -> boxes/classes/scores/masks.

    Returns dict with:
        boxes_xyxy  (N,4) float, in 640x640 input pixels
        classes     (N,)  int
        scores      (N,)  float
        masks       (N,640,640) float 0..1 (threshold at ~0.5 downstream)
    """
    endnodes = order_endnodes(outputs)
    strides = tuple(reversed(V8_STRIDES))  # endnodes go 20->40->80
    image_dims = V8_INPUT_SHAPE

    raw_boxes = endnodes[:7:3]
    scores = [
        np.reshape(s, (-1, s.shape[1] * s.shape[2], V8_NUM_CLASSES)) for s in endnodes[1:8:3]
    ]
    scores = np.concatenate(scores, axis=1)
    decoded_boxes = _yolov8_decoding(raw_boxes, strides, image_dims, V8_REG_MAX)

    proto_data = endnodes[9]
    n_masks = proto_data.shape[3]
    fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1))
    scores_obj = np.concatenate([fake_objectness, scores], axis=-1)
    coeffs = [
        np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks)) for c in endnodes[2:9:3]
    ]
    coeffs = np.concatenate(coeffs, axis=1)

    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
    nms_res = non_max_suppression(
        predictions,
        conf_thres=score_threshold,
        iou_thres=nms_iou,
        max_det=max_det,
        nm=n_masks,
        multi_label=False,
    )[0]

    masks = process_mask(
        proto_data[0].astype(np.float32, copy=False),
        nms_res["mask"],
        nms_res["detection_boxes"],
        image_dims,
    )
    return {
        "boxes_xyxy": nms_res["detection_boxes"],
        "classes": nms_res["detection_classes"].astype(int),
        "scores": nms_res["detection_scores"],
        "masks": masks,
    }
