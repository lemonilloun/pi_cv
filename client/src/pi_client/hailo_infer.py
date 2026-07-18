"""Thin multi-model wrapper over pyhailort for the scene recorder.

Runs several hefs (yolov8-seg + CLIP image encoder) on one Hailo-8 through
the model scheduler: a single VDevice with HailoSchedulingAlgorithm lets
configured models time-share the NPU — exactly what a 2-3 fps keyframe
pipeline needs (picamera2.devices.Hailo can't do this: it owns the device
for one hef).

Uses the InferVStreams (sync) API — stable across HailoRT 4.x.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class HailoMultiModel:
    def __init__(self) -> None:
        from hailo_platform import VDevice, HailoSchedulingAlgorithm

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)
        self._models: dict[str, "_ConfiguredHef"] = {}

    def load(self, name: str, hef_path: str) -> "_ConfiguredHef":
        model = _ConfiguredHef(self._vdevice, hef_path)
        self._models[name] = model
        logger.info(
            "Hailo model '%s' ready: input %s", name, model.input_shape
        )
        return model

    def close(self) -> None:
        for model in self._models.values():
            model.close()
        try:
            self._vdevice.release()
        except Exception:
            pass


class _ConfiguredHef:
    def __init__(self, vdevice: Any, hef_path: str) -> None:
        from hailo_platform import (
            HEF,
            ConfigureParams,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            FormatType,
        )

        self.hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        network_group = vdevice.configure(self.hef, configure_params)[0]
        self._network_group = network_group
        self._input_params = InputVStreamParams.make_from_network_group(
            network_group, format_type=FormatType.UINT8
        )
        self._output_params = OutputVStreamParams.make_from_network_group(
            network_group, format_type=FormatType.FLOAT32
        )
        self._infer_cls = InferVStreams

        info = self.hef.get_input_vstream_infos()[0]
        self.input_name = info.name
        self.input_shape = tuple(info.shape)  # (H, W, C)

    def infer(self, image_hwc_uint8: np.ndarray) -> dict[str, np.ndarray]:
        """Run one frame; returns {output_name: array} with batch dim
        stripped."""
        batch = np.expand_dims(image_hwc_uint8, axis=0)
        with self._infer_cls(
            self._network_group, self._input_params, self._output_params
        ) as pipeline:
            results = pipeline.infer({self.input_name: batch})
        out: dict[str, np.ndarray] = {}
        for name, arr in results.items():
            arr = np.asarray(arr)
            out[name] = arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr
        return out

    def close(self) -> None:
        pass  # network groups are released with the vdevice
