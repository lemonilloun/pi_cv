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

import dataclasses
from typing import Any

import numpy as np


V8_STRIDES = (8, 16, 32)
V8_REG_MAX = 15
V8_NUM_CLASSES = 80
V8_INPUT_SHAPE = (640, 640)
V8_SCORE_THRESHOLD = 0.001
V8_NMS_IOU = 0.7


@dataclasses.dataclass(frozen=True)
class SegArch:
    """The shape facts a YOLOv8-seg-family decode needs, in one object.

    These were constants because there was one model. FastSAM is the same
    architecture with a single class, so the decode is unchanged and only these
    numbers move — but they must move *together*, and two of them were
    previously baked into `order_endnodes` as literals (`h == 160`, the
    `{64, 80, 32}` role map).

    `input_shape` is the one worth being loud about: the recorder reads the real
    hef input shape at load time, while this module assumed 640x640. A hef with
    a different input decoded silently into the wrong coordinate space — wrong
    boxes, wrong masks, no error. Build the arch from the measured shape
    (`dataclasses.replace(YOLOV8_SEG, input_shape=model.input_shape[:2])`)
    rather than trusting the default.

    Verify a hef against one of these with `scripts/probe_hef.py`, which derives
    all four numbers from the output blobs without being told them.
    """

    name: str
    num_classes: int
    reg_max: int = V8_REG_MAX
    num_masks: int = 32
    input_shape: tuple[int, int] = V8_INPUT_SHAPE
    strides: tuple[int, ...] = V8_STRIDES
    # Whether the class head returns raw logits that still need a sigmoid.
    #
    # This is NOT a property of the architecture but of where the hef's graph
    # was cut, and getting it wrong is silent: logits and probabilities are
    # both float blobs of the right shape, so the decode runs happily and
    # compares something like -34.5 against a 0.4 threshold, finding nothing
    # forever. Measured with scripts/probe_hef.py on a real frame — the Model
    # Zoo hefs report 0.000..0.161 (already sigmoid) while our DFC build, cut
    # at `/model.23/cv3.x/cv3.x.2/Conv`, reports -45.7..1.48.
    class_logits: bool = False

    def __post_init__(self) -> None:
        roles = (self.box_channels, self.num_classes, self.num_masks)
        if len(set(roles)) != len(roles):
            # Output blobs are told apart by channel count and nothing else, so
            # two roles of equal width are genuinely indistinguishable here.
            # Refusing beats picking: swapped class scores and mask
            # coefficients produce masks that look like masks and belong to the
            # wrong objects.
            raise ValueError(
                f"SegArch {self.name!r} is undecodable: boxes/classes/masks have "
                f"channel counts {roles}, which are not all distinct")

    @property
    def box_channels(self) -> int:
        """DFL box head width: 4 sides x (reg_max + 1) distribution bins."""
        return 4 * (self.reg_max + 1)

    @property
    def scale_heights(self) -> tuple[int, ...]:
        """Detection grid heights, ascending — i.e. by descending stride."""
        return tuple(sorted(self.input_shape[0] // s for s in self.strides))


YOLOV8_SEG = SegArch(name="yolov8_seg", num_classes=V8_NUM_CLASSES)

# FastSAM is YOLOv8-seg with a single, meaningless class: every proposal is
# "object". Confirmed by shape in scripts/probe_hef.py before use — do not take
# this on faith from a forum post.
FASTSAM_S = SegArch(name="fastsam_s", num_classes=1)

# The fine-tune: yolo11l-seg on 31 indoor ADE20K classes. YOLO11's head is
# geometrically identical to YOLOv8's — same Segment module, reg_max 16 (=> 64
# box channels), 32 mask coefficients — which is why this decoder needs nothing
# but a different class count. Verified twice: on the ultralytics model before
# training, and on the exported ONNX, whose ten head outputs are 64/31/32 per
# scale plus 32-channel prototypes.
#
# The class NAMES live in config/seg_classes_indoor.json, not here: the order
# of that file is baked into the weights, and duplicating it in code is how the
# two drift apart and every label silently shifts by one.
INDOOR_ADE20K = SegArch(name="indoor_ade20k", num_classes=31, class_logits=True)

SEG_ARCHS = {arch.name: arch for arch in (YOLOV8_SEG, FASTSAM_S, INDOOR_ADE20K)}


@dataclasses.dataclass(frozen=True)
class Letterbox:
    """How a frame was fitted into the model's square input, and how to undo it.

    Every YOLO checkpoint in this chain — the Model Zoo hefs, the ultralytics
    weights we fine-tune from, and the fine-tune itself — is trained and
    validated on letterboxed input: aspect ratio preserved, the leftover
    padded with grey 114. The recorder used to `cv2.resize` a 1536x864 frame
    straight into 640x640, squashing 16:9 into a square, which is a shape the
    model has never seen.

    Measured on the 53 recorded keyframes with the stock COCO hef: squashing
    found 11 detections, letterboxing found 16, and on one keyframe the same
    object moved 0.369 -> 0.494, i.e. from under the 0.4 threshold to over it.

    `scale`, `pad_x`, `pad_y` are kept so detections can be mapped back: the
    model returns coordinates in padded space and everything downstream —
    boxes, masks, the depth lookup — works in frame pixels.
    """

    scale: float
    pad_x: int
    pad_y: int
    content_w: int
    content_h: int
    input_w: int
    input_h: int

    def box_to_frame(self, box_xyxy: np.ndarray) -> np.ndarray:
        """(N,4) in model input pixels -> (N,4) in original frame pixels."""
        out = np.asarray(box_xyxy, dtype=np.float64).copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale
        return out

    def crop_masks(self, masks: np.ndarray) -> np.ndarray:
        """Drop the padding band from (N, H, W) model-space masks.

        After this the mask grid is proportional to the frame again, so every
        reader can keep scaling masks to whatever resolution it needs without
        knowing letterboxing happened at all.
        """
        if masks is None or masks.shape[0] == 0:
            return masks
        return masks[
            :,
            self.pad_y:self.pad_y + self.content_h,
            self.pad_x:self.pad_x + self.content_w,
        ]


def letterbox(image: np.ndarray, input_shape: tuple[int, int],
              pad_value: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize preserving aspect ratio, centre, pad with grey. Returns the
    padded image and the transform needed to undo it."""
    import cv2

    input_h, input_w = int(input_shape[0]), int(input_shape[1])
    height, width = image.shape[:2]
    scale = min(input_w / width, input_h / height)
    content_w = max(1, int(round(width * scale)))
    content_h = max(1, int(round(height * scale)))
    pad_x = (input_w - content_w) // 2
    pad_y = (input_h - content_h) // 2

    canvas = np.full((input_h, input_w, image.shape[2]), pad_value, dtype=image.dtype)
    canvas[pad_y:pad_y + content_h, pad_x:pad_x + content_w] = cv2.resize(
        image, (content_w, content_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, Letterbox(
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        content_w=content_w, content_h=content_h,
        input_w=input_w, input_h=input_h,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic.

    The naive `1/(1+exp(-x))` overflows for strongly negative x, which used to
    be unreachable: the Model Zoo hefs emit already-sigmoid class scores in
    0..1. Our own DFC build stops at the class convolution and emits logits
    that reach -45 on a real frame, and `exp(45)` warns. The result was still
    correct (it saturates to 0, which is the right answer) but a RuntimeWarning
    on every frame trains you to ignore warnings.
    """
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


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


def order_endnodes(
    outputs: dict[str, np.ndarray], arch: SegArch = YOLOV8_SEG
) -> list[np.ndarray]:
    """Arrange the hef's 10 raw output tensors into the order the decoder
    expects: per scale, coarsest grid first, the (boxes, scores, coeffs)
    triple, then the prototypes.

    Identified by shape, so output layer names don't matter — and checked
    against `arch`, so passing the wrong one is an error rather than a silent
    misread. Prototypes are the blob whose channel count is `num_masks` at a
    grid finer than any detection scale (input/4 vs input/8 at the coarsest),
    which replaces the old `h == 160` literal.
    """
    protos = None
    by_scale: dict[int, dict[str, np.ndarray]] = {}
    scale_heights = arch.scale_heights
    role_by_channels = {
        arch.box_channels: "boxes",
        arch.num_classes: "scores",
        arch.num_masks: "coeffs",
    }
    for name, arr in outputs.items():
        if arr.ndim == 3:  # (H, W, C) — add batch dim
            arr = arr[np.newaxis, ...]
        _, h, w, c = arr.shape
        if c == arch.num_masks and h not in scale_heights:
            if protos is not None:
                raise ValueError(
                    f"Two prototype-shaped outputs for arch {arch.name!r}; "
                    f"second is {name} {arr.shape}")
            protos = arr
            continue
        role = role_by_channels.get(c)
        if role is None:
            raise ValueError(
                f"Unexpected seg output {name} with shape {arr.shape} for arch "
                f"{arch.name!r} (expected channel counts {sorted(role_by_channels)})")
        by_scale.setdefault(h, {})[role] = arr
    if protos is None or sorted(by_scale) != list(scale_heights):
        raise ValueError(
            f"Unexpected seg output set for arch {arch.name!r}: "
            f"scales={sorted(by_scale)} (expected {list(scale_heights)}), "
            f"protos={protos is not None}")
    endnodes = []
    for scale in scale_heights:
        triple = by_scale[scale]
        missing = {"boxes", "scores", "coeffs"} - set(triple)
        if missing:
            raise ValueError(
                f"Scale {scale} is missing {sorted(missing)} for arch {arch.name!r}")
        endnodes += [triple["boxes"], triple["scores"], triple["coeffs"]]
    endnodes.append(protos)
    return endnodes


def yolov8_seg_postprocess(
    outputs: dict[str, np.ndarray],
    score_threshold: float = 0.25,
    nms_iou: float = V8_NMS_IOU,
    max_det: int = 50,
    arch: SegArch = YOLOV8_SEG,
) -> dict[str, Any]:
    """Full decode: raw hef outputs -> boxes/classes/scores/masks.

    Returns dict with:
        boxes_xyxy  (N,4) float, in arch.input_shape pixels
        classes     (N,)  int  (all zero for a single-class arch like FastSAM)
        scores      (N,)  float
        masks       (N,) + arch.input_shape, float 0..1 (threshold ~0.5 later)
    """
    endnodes = order_endnodes(outputs, arch)
    # endnodes run coarsest grid first, which is largest stride first.
    strides = tuple(sorted(arch.strides, reverse=True))
    image_dims = arch.input_shape

    raw_boxes = endnodes[:7:3]
    scores = [
        np.reshape(s, (-1, s.shape[1] * s.shape[2], arch.num_classes))
        for s in endnodes[1:8:3]
    ]
    scores = np.concatenate(scores, axis=1)
    if arch.class_logits:
        # The hef stopped at the class convolution, so the sigmoid YOLO applies
        # after it has to happen here. Without this every score is a logit and
        # the threshold comparison downstream is meaningless.
        scores = _sigmoid(scores)
    decoded_boxes = _yolov8_decoding(raw_boxes, strides, image_dims, arch.reg_max)

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
