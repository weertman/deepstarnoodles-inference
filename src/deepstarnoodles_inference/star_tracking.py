# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Post-hoc star identities for records annotated frame by frame at low frame rates (1-2 fps).

``track-videos`` uses this tracker by default. BoT-SORT, which live replay uses, matches boxes on IoU and confirms a
new track over one or two frames; at 1-2 fps a star moves about a body width between frames, so BoT-SORT splits one
star into many IDs and holds back each new track's first detection. This tracker is for the offline pass only.

Per record it solves one global assignment (``scipy.optimize.linear_sum_assignment``) between the live tracks and the
record's detections on

    cost = d + size_weight * |ln(s_det / s_track)|

where ``s = sqrt(box area)`` and ``d`` is the distance between box centres divided by ``sqrt(s_track * s_det)``. A
track's size is the median of its last ``size_hist`` observed sizes; its position is its last observed box centre
(no motion model). A pair is allowed only when ``d <= min(gate + gate_per_s * dt, gate_max)``, with ``dt`` the
seconds since the track was last seen, and the two sizes differ by at most a factor ``size_gate``. Time is the
record's ``time_s`` (or ``frame / fps``), so missing records widen the gate like any other gap. A track unseen for
more than ``max_gap_s`` seconds ends.

Every detection that is not matched starts a new ID at once (no confirmation delay); IDs are numbered from 1 in
order of birth. Two boxes of one record with IoU >= ``dup_iou`` are the same star detected twice: the less confident
one gets no ID (``None``; ``track-videos`` moves it to ``untracked_stars``). ``min_conf_new`` is a confidence floor
for starting a track (default 0: off).

Deterministic: no randomness; tracks are kept in ID order and detections in record order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e9


@dataclass(frozen=True)
class HungarianParams:
    """Settings of :class:`HungarianStarTracker` (the defaults are the released ``track-videos`` settings)."""

    gate: float = 2.0          # max normalised centre distance at dt -> 0
    gate_per_s: float = 1.0    # extra normalised distance allowed per second since the track was last seen
    gate_max: float = 4.0      # cap on the gate
    size_weight: float = 1.0   # weight of |ln size ratio| in the cost
    size_gate: float = 2.0     # max size ratio, either way
    size_hist: int = 3         # a track's size is the median of its last size_hist observed sizes
    max_gap_s: float = 60.0    # a track unseen for longer than this ends
    dup_iou: float = 0.95      # same-record boxes with IoU >= this: the less confident one is a duplicate (0 = off)
    min_conf_new: float = 0.0  # detections below this confidence start no track

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"star-tracker parameter {item.name} must be a finite number, got {value!r}")
        if not 0 < self.gate <= self.gate_max:
            raise ValueError("star-tracker parameters need 0 < gate <= gate_max")
        if self.gate_per_s < 0 or self.size_weight < 0:
            raise ValueError("star-tracker parameters gate_per_s and size_weight must be non-negative")
        if not self.size_gate > 1:
            raise ValueError("star-tracker parameter size_gate must be greater than 1")
        if not isinstance(self.size_hist, int) or self.size_hist < 1:
            raise ValueError("star-tracker parameter size_hist must be a positive integer")
        if not self.max_gap_s > 0:
            raise ValueError("star-tracker parameter max_gap_s must be positive")
        if not 0 <= self.dup_iou <= 1:
            raise ValueError("star-tracker parameter dup_iou must be between 0 and 1")
        if not 0 <= self.min_conf_new <= 1:
            raise ValueError("star-tracker parameter min_conf_new must be between 0 and 1")

    @classmethod
    def from_mapping(cls, values: dict | None) -> "HungarianParams":
        """Parameters from ``{name: value}`` (values may be strings, as ``--star-param`` gives them)."""
        values = dict(values or {})
        known = {item.name: item for item in fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown star-tracker parameters: {', '.join(unknown)} "
                             f"(known: {', '.join(known)})")
        converted = {}
        for name, value in values.items():
            kind = int if known[name].type in (int, "int") else float
            try:
                number = float(value)
                if kind is int:
                    if not number.is_integer():
                        raise ValueError
                    number = int(number)
            except (TypeError, ValueError):
                raise ValueError(f"star-tracker parameter {name} must be {'an integer' if kind is int else 'a number'}"
                                 f", got {value!r}") from None
            converted[name] = number
        return cls(**converted)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Track:
    track_id: int
    centre: np.ndarray
    time_s: float
    sizes: list = field(default_factory=list)
    size: float = 0.0


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


class HungarianStarTracker:
    """Online over one video's records in time order; call :meth:`update` once per record.

    After each call, :attr:`ended` lists the IDs that ended (unseen for more than ``max_gap_s``) before that record,
    and :attr:`live_ids` the IDs still open.
    """

    def __init__(self, params: HungarianParams | None = None) -> None:
        self.params = params or HungarianParams()
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.ended: list[int] = []
        self._last_time: float | None = None

    @property
    def live_ids(self) -> list[int]:
        return [track.track_id for track in self.tracks]

    def update(self, stars: list[dict], time_s: float) -> list[int | None]:
        """IDs for one record's ``stars`` (each with ``crop.bbox_xyxy`` and ``crop.confidence``), in their order.

        ``None`` marks a duplicate box or one below ``min_conf_new``. ``time_s`` must not decrease.
        """
        p = self.params
        t = float(time_s)
        if not math.isfinite(t):
            raise ValueError(f"record time must be finite, got {time_s!r}")
        if self._last_time is not None and t < self._last_time:
            raise ValueError(f"record times must not decrease; {t} came after {self._last_time}")
        self._last_time = t
        self.ended = [track.track_id for track in self.tracks if t - track.time_s > p.max_gap_s]
        if self.ended:
            self.tracks = [track for track in self.tracks if t - track.time_s <= p.max_gap_s]
        n = len(stars)
        ids: list[int | None] = [None] * n
        if n == 0:
            return ids
        boxes = np.asarray([star["crop"]["bbox_xyxy"] for star in stars], dtype=np.float64).reshape(-1, 4)
        conf = np.asarray([float(star["crop"]["confidence"]) for star in stars], dtype=np.float64)
        if not (np.all(np.isfinite(boxes)) and np.all(np.isfinite(conf))):
            raise ValueError("star boxes and confidences must be finite")
        size = np.sqrt(np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None) * np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None))
        centre = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        active = np.ones(n, dtype=bool)
        if p.dup_iou > 0 and n > 1:
            iou = _iou_matrix(boxes)
            order = sorted(range(n), key=lambda i: (-conf[i], i))
            for rank, i in enumerate(order):
                if active[i]:
                    for j in order[rank + 1:]:
                        if active[j] and iou[i, j] >= p.dup_iou:
                            active[j] = False
        detections = np.flatnonzero(active)
        matched: set[int] = set()
        if self.tracks and len(detections):
            track_centre = np.stack([track.centre for track in self.tracks])
            track_size = np.asarray([track.size for track in self.tracks])
            dt = np.asarray([max(t - track.time_s, 0.0) for track in self.tracks])
            gate = np.minimum(p.gate + p.gate_per_s * dt, p.gate_max)
            distance = np.linalg.norm(track_centre[:, None, :] - centre[None, detections, :], axis=2)
            normalised = distance / np.sqrt(track_size[:, None] * size[None, detections])
            log_ratio = np.abs(np.log(size[None, detections] / track_size[:, None]))
            allowed = (normalised <= gate[:, None]) & (log_ratio <= math.log(p.size_gate))
            cost = np.where(allowed, normalised + p.size_weight * log_ratio, _BIG)
            for row, column in zip(*linear_sum_assignment(cost)):
                if not allowed[row, column]:
                    continue
                i = int(detections[column])
                track = self.tracks[row]
                track.centre, track.time_s = centre[i], t
                track.sizes = (track.sizes + [float(size[i])])[-p.size_hist:]
                track.size = float(np.median(track.sizes))
                ids[i] = track.track_id
                matched.add(i)
        for i in detections.tolist():
            if i in matched or conf[i] < p.min_conf_new:
                continue
            track = _Track(self.next_id, centre[i], t, sizes=[float(size[i])], size=float(size[i]))
            self.next_id += 1
            self.tracks.append(track)
            ids[i] = track.track_id
        return ids
