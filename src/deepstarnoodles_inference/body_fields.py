# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Per-star body origin, support area and support centroid, summarized on the device.

The values equal, bit for bit, the host rules they replace:

* body origin: :func:`runtime.body_center_from_maps` (= DeepStarNoodles ``runtime.body_center``):
  the first mouth peak (``find_peaks(mouth, 0.15, 8)``), else the mean of ``support > 0.3`` pixels,
  else the crop centre; code 0, 1 or 2 (``noodle_polylines`` ``BODY_ORIGIN_CODES``);
* support area: the number of ``support > 0.5`` pixels; support centroid: their float64 mean
  ``[x, y]``, ``None`` when the area is 0 (the paper re-trace's frame landmarks).

``find_peaks(...)[0]`` is the global maximum of the mouth map whenever that maximum is at least
``f32(0.15)``: it always passes the max-pool test, the output is sorted by score, and NMS never drops
the first peak. The device therefore needs one max/argmax, plus a count of pixels equal to the maximum.
When several pixels tie for it, or it is not finite, numpy's unstable ``argsort`` decides, so that crop
falls back to the host rule on the very same maps. Pixel counts and index sums are exact integers
(below 2**53), so their float64 quotient equals numpy's ``mean`` exactly.

Everything runs outside ``torch.compile`` and autocast on the eager float32 ``sigmoid`` of the raw
Hydra output (or the activated maps already computed from it), and costs one small device-to-host copy
per crop group, taken after the GRU has drained the stream.
"""
from __future__ import annotations

import threading

import numpy as np
from scipy.ndimage import maximum_filter
import torch

from .records import (AREA_SUPPORT_THRESHOLD, BODY_ORIGIN_CENTRE, BODY_ORIGIN_MOUTH, BODY_ORIGIN_SUPPORT,
                      MOUTH_PEAK_NMS, MOUTH_PEAK_THRESHOLD, ORIGIN_SUPPORT_THRESHOLD)
_INDEX_LOCK = threading.Lock()
_INDEX: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_MOUTH_THRESHOLD_F32 = np.float32(MOUTH_PEAK_THRESHOLD)


def _indices(device: torch.device, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached int64 row and column index vectors; created once, never inside graph capture."""
    key = (str(device), int(height), int(width))
    with _INDEX_LOCK:
        cached = _INDEX.get(key)
        if cached is None:
            cached = (torch.arange(height, dtype=torch.int64, device=device),
                      torch.arange(width, dtype=torch.int64, device=device))
            if device.type == "cuda":
                # Other streams (crop-stream workspaces) use these later; finish creating them first.
                torch.cuda.current_stream(device).synchronize()
            _INDEX[key] = cached
        return cached


def first_peak(heat: np.ndarray, threshold: float, nms_radius: int) -> np.ndarray | None:
    """``decode.find_peaks(heat, threshold, nms_radius)[0, :2]`` (or None), without the NMS loop.

    The same candidates, scores and ``argsort`` as ``find_peaks``, whose first sorted candidate NMS
    always keeps. A saturated plateau makes every plateau pixel a candidate, and the full NMS loop is
    then quadratic in the plateau size; this stays O(n log n).
    """
    k = 2 * int(nms_radius) + 1
    pooled = maximum_filter(np.asarray(heat, dtype=np.float32), size=k, mode="constant", cval=-np.inf)
    ys, xs = np.nonzero((heat >= pooled - 1e-6) & (heat >= threshold))
    if len(xs) == 0:
        return None
    first = np.argsort(-heat[ys, xs])[0]
    return np.stack([xs[[first]], ys[[first]], heat[ys, xs][[first]]], axis=1).astype(np.float32)[0, :2]


def origin_from_maps(support_map: np.ndarray, mouth_map: np.ndarray) -> tuple[np.ndarray, int]:
    """Host rule: ``(runtime.body_center_from_maps(support_map, mouth_map), source code)``."""
    peak = first_peak(mouth_map, MOUTH_PEAK_THRESHOLD, MOUTH_PEAK_NMS)
    if peak is not None:
        return peak, BODY_ORIGIN_MOUTH
    ys, xs = np.nonzero(support_map > ORIGIN_SUPPORT_THRESHOLD)
    if len(xs):
        return np.asarray([xs.mean(), ys.mean()], dtype=np.float32), BODY_ORIGIN_SUPPORT
    height, width = support_map.shape[-2:]
    return np.asarray([width / 2.0, height / 2.0], dtype=np.float32), BODY_ORIGIN_CENTRE


def host_body_fields(support_map: np.ndarray, mouth_map: np.ndarray) -> dict:
    """The four star fields computed on the host from one crop's maps (reference implementation)."""
    origin, code = origin_from_maps(support_map, mouth_map)
    ys, xs = np.nonzero(support_map > AREA_SUPPORT_THRESHOLD)
    centroid = [float(xs.mean()), float(ys.mean())] if len(xs) else None
    return {"body_origin_model": [float(origin[0]), float(origin[1])], "body_origin_code": int(code),
            "support_area_model_px": int(len(xs)), "support_centroid_model": centroid}


class PendingBody:
    """Device summary of one crop group; :meth:`finish` copies it once and builds the star fields."""

    def __init__(self, packed: torch.Tensor, height: int, width: int, maps_of) -> None:
        self.packed, self.height, self.width, self.maps_of = packed, height, width, maps_of

    def finish(self) -> list[dict]:
        host = self.packed.cpu().numpy()
        self.packed = None
        out = []
        for b, (mmax, arg, ties, n5, sx5, sy5, n3, sx3, sy3) in enumerate(host.tolist()):
            if n5:
                area, centroid = int(n5), [sx5 / n5, sy5 / n5]
            else:
                area, centroid = 0, None
            peak = np.isfinite(mmax) and np.float32(mmax) >= _MOUTH_THRESHOLD_F32
            if not np.isfinite(mmax) or (peak and ties != 1):
                # Tied or non-finite maximum: numpy's argsort decides, so use the host rule on these maps.
                support_map, mouth_map = self.maps_of(b)
                origin, code = origin_from_maps(support_map, mouth_map)
            elif peak:
                arg = int(arg)
                origin, code = (np.float32(arg % self.width), np.float32(arg // self.width)), BODY_ORIGIN_MOUTH
            elif n3:
                origin, code = (np.float32(sx3 / n3), np.float32(sy3 / n3)), BODY_ORIGIN_SUPPORT
            else:
                origin = (np.float32(self.width / 2.0), np.float32(self.height / 2.0))
                code = BODY_ORIGIN_CENTRE
            out.append({"body_origin_model": [float(origin[0]), float(origin[1])], "body_origin_code": int(code),
                        "support_area_model_px": area, "support_centroid_model": centroid})
        return out


@torch.no_grad()
def body_summary(support: torch.Tensor, mouth: torch.Tensor, maps_of) -> PendingBody:
    """Queue the summary of ``(B,H,W)`` float32 support and mouth maps on the current stream (no sync).

    ``maps_of(b)`` returns host ``(support, mouth)`` arrays of crop ``b`` equal to these maps; it is
    called only for crops whose mouth maximum is tied or not finite.
    """
    if support.ndim != 3 or support.shape != mouth.shape:
        raise ValueError("support and mouth maps must be matching (B,H,W) tensors")
    if support.dtype != torch.float32 or mouth.dtype != torch.float32:
        raise ValueError("body summary requires float32 activated maps")
    _, height, width = support.shape
    iy, ix = _indices(support.device, height, width)
    flat = mouth.reshape(len(mouth), -1)
    mmax, arg = flat.max(dim=1)
    ties = (flat == mmax[:, None]).sum(dim=1)
    columns = [mmax.double(), arg.double(), ties.double()]
    for threshold in (AREA_SUPPORT_THRESHOLD, ORIGIN_SUPPORT_THRESHOLD):
        # A Python scalar threshold compares in float32, as numpy does.
        mask = support > threshold
        rows, cols = mask.sum(dim=2), mask.sum(dim=1)
        columns += [rows.sum(dim=1).double(), (cols * ix).sum(dim=1).double(), (rows * iy).sum(dim=1).double()]
    return PendingBody(torch.stack(columns, dim=1), height, width, maps_of)


def maps_for_body(raw: torch.Tensor, activated: torch.Tensor | None):
    """``(support, mouth, maps_of)`` for :func:`body_summary` from one Hydra output.

    ``activated`` is None (archive minimal: activate the two channels of ``raw``), compact live maps
    ``(B,2,H,W)`` (support, mouth), or full activated fields ``(B,10+D,H,W)`` (support 0, mouth 5).
    """
    if activated is None:
        def maps_of(b):
            # The same kernel on the same tensor as below, so the maps are bit-identical.
            return (torch.sigmoid(raw[:, 0])[b].cpu().numpy(), torch.sigmoid(raw[:, 5])[b].cpu().numpy())
        return torch.sigmoid(raw[:, 0]), torch.sigmoid(raw[:, 5]), maps_of
    mouth_channel = 1 if activated.shape[1] == 2 else 5

    def maps_of(b):
        return activated[b, 0].cpu().numpy(), activated[b, mouth_channel].cpu().numpy()
    return activated[:, 0], activated[:, mouth_channel], maps_of


def summarize(raw: torch.Tensor, activated: torch.Tensor | None) -> PendingBody:
    """Queue the body summary of one Hydra output; the transient maps are released on return."""
    support, mouth, maps_of = maps_for_body(raw, activated)
    return body_summary(support, mouth, maps_of)
