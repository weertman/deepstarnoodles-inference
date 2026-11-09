# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Star and arm identities for videos that were annotated frame by frame.

``serve-videos`` annotates every frame on its own, so its records number arms per frame (``arm_index``) and leave
``star_track_id`` empty. ``track-videos`` gives each star an ID across the video, then runs one arm tracker per star,
which measures arms from that star's saved body origin (``body_origin_model``). It needs no GPU, model weights or
video. Records from before 0.5.0.dev11 have no body origin and can't be tracked.

Star IDs come from the post-hoc Hungarian tracker (default; ``star_tracking``), built for 1-2 fps records: it matches
on size-normalised centre distance with a gate that grows with the time since a star was seen, bridges gaps and
missing records up to 60 s, and gives every star an ID in the record it first appears in. ``--star-tracker botsort``
restores the 0.5.0.dev12 behaviour byte for byte: BoT-SORT as live replay runs it (without camera-motion
compensation, which needs the frames), which at 1-2 fps splits a moving star into many IDs and holds each new star's
first detection back in ``untracked_stars``. Arm IDs come from the released arm tracker either way.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from ._vendor.noodle_polylines import _assoc
from ._vendor.noodle_polylines.geometry import resample_tip
from ._vendor.noodle_polylines.tracking import RELEASED, TrackerConfig
from .paths import DEFAULT_BOTSORT_CONFIG
from .runtime import DEFAULT_STAR_TRACK_MAX_AGE, K_TRACK
from .star_tracking import HungarianParams, HungarianStarTracker

SUFFIXES = (".jsonl", ".jsonl.zst")


class _Boxes:
    """The part of Ultralytics' ``Boxes`` that ``BYTETracker.update`` reads."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "_Boxes":
        return _Boxes(self.xyxy[index], self.conf[index], self.cls[index])


class ArmTrackers:
    """One released arm tracker per star ID, as :class:`runtime.StarArmTrackerRegistry` keeps for live replay.

    Each star's arms go through the vendored ``noodle_polylines`` array-state FusedTracker with the released settings
    (shape gate 120 px, embedding weight 2.0, max age 16 arm-bearing updates, match-cost cap 90, k = 12), which
    gives the arm IDs the package's ``fused_tracker`` gives, bit for bit, about 40 times faster. A star whose record
    has no arms does not step its tracker, so missing arms do not age arm tracks.
    """

    def __init__(self, config: TrackerConfig = RELEASED) -> None:
        if config.vel_alpha != 0 or config.k != K_TRACK:
            raise ValueError("the array tracker reproduces the released settings only (vel_alpha 0, k 12)")
        self.config = config
        self._trackers: dict[int, _assoc.ArrayTracker] = {}
        self._last_seen: dict[int, int] = {}

    @property
    def star_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._trackers))

    def track(self, star_track_id: int, frame: int, arms: list[dict], center) -> list[int]:
        """Arm IDs for one star's arms (``polyline_crop``, ``identity``) measured from its body ``center``."""
        star_track_id = int(star_track_id)
        tracker = self._trackers.get(star_track_id)
        if tracker is None:
            tracker = self._trackers[star_track_id] = _assoc.ArrayTracker(self.config, self.config.k)
        self._last_seen[star_track_id] = int(frame)
        if not arms:
            return []
        polylines = [arm["polyline_crop"] for arm in arms]
        counts = [len(polyline) for polyline in polylines]
        points = np.asarray([point for polyline in polylines for point in polyline], dtype=np.float32)
        if min(counts) < 1 or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"frame {frame}: every arm polyline_crop must be a non-empty list of [x, y] points")
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        origin = np.broadcast_to(np.asarray(center, dtype=np.float32).reshape(2), (len(arms), 2))
        sampled, lengths = resample_tip(points, offsets, self.config.k, arm_origin=origin)
        lengths = lengths.astype(np.float32)
        embeds = np.asarray([arm["identity"] for arm in arms], dtype=np.float32)
        if embeds.ndim != 2 or not _assoc.fast_path_ok(self.config, sampled, lengths, embeds,
                                                         np.arange(len(arms))):
            raise ValueError(f"frame {frame}: arm polylines, identities and the body origin must be finite")
        rows = _assoc.arm_rows(sampled, lengths, self.config.k)
        return tracker.step(rows, embeds, sampled, lengths).tolist()

    def discard(self, star_track_id: int) -> None:
        """Forget one star's arm tracker (its star track ended)."""
        self._trackers.pop(int(star_track_id), None)
        self._last_seen.pop(int(star_track_id), None)

    def prune(self, frame: int, max_age: int = DEFAULT_STAR_TRACK_MAX_AGE) -> None:
        """Forget stars unseen for more than ``max_age`` frames (live replay's BoT-SORT horizon)."""
        for star_track_id in [key for key, seen in self._last_seen.items() if int(frame) - seen > max_age]:
            self.discard(star_track_id)


def star_tracker(config: Path = DEFAULT_BOTSORT_CONFIG):
    """A fresh BoT-SORT with the configuration live replay uses."""
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace

    return BOTSORT(args=IterableSimpleNamespace(**YAML.load(str(config))))


STAR_TRACKERS = ("hungarian", "botsort")
RUN_LOG = "_track-videos.jsonl"   # one line per video tracked into an --output directory


def _frames_in_order(records: Iterable[dict]) -> Iterator[tuple[int, dict]]:
    previous = None
    for record in records:
        frame = int(record["frame"])
        if previous is not None and frame <= previous:
            raise ValueError(f"frames must increase within a video; frame {frame} came after {previous}")
        previous = frame
        yield frame, record


def _record_time(record: dict, frame: int) -> float:
    """Seconds into the video: ``time_s``, else ``frame / fps``, else the frame number."""
    if record.get("time_s") is not None:
        return float(record["time_s"])
    if record.get("fps"):
        return frame / float(record["fps"])
    return float(frame)


def _split(registry: ArmTrackers, frame: int, stars: list[dict], star_ids: list[int | None]) -> tuple[list, list]:
    """Set star and arm IDs; ``(tracked, untracked)`` stars in record order."""
    tracked, untracked = [], []
    for star, star_id in zip(stars, star_ids):
        star["star_track_id"] = star_id
        if star_id is None:
            for arm in star["arms"]:
                arm["track_id"] = None
            untracked.append(star)
            continue
        if "body_origin_model" not in star:
            raise ValueError("these records have no body origin; annotate the video with "
                             "deepstarnoodles-inference 0.5.0.dev11 or later")
        labels = registry.track(star_id, frame, star["arms"], star["body_origin_model"])
        for arm, label in zip(star["arms"], labels):
            arm["track_id"] = int(label)
        tracked.append(star)
    return tracked, untracked


def _track_records_botsort(records: Iterable[dict], tracker_config: Path = DEFAULT_BOTSORT_CONFIG) -> Iterator[dict]:
    """The 0.5.0.dev12 ``track-videos``: BoT-SORT star IDs as live replay assigns them."""
    from ultralytics.trackers.basetrack import BaseTrack

    tracker = star_tracker(tracker_config)
    registry = ArmTrackers()
    # Live replay numbers each video's stars from 1 (crop.TrackedStarCropper); this runs one video at a time.
    BaseTrack.reset_id()
    for frame, record in _frames_in_order(records):
        stars = record.get("stars") or []
        # The class column carries each box's record index: BoT-SORT does not use classes, and the rows' last column
        # (detection index) counts within the high- or low-confidence subset in ultralytics 8.4.51, not the record.
        boxes = _Boxes(np.asarray([star["crop"]["bbox_xyxy"] for star in stars], dtype=np.float32).reshape(-1, 4),
                       np.asarray([star["crop"]["confidence"] for star in stars], dtype=np.float32),
                       np.arange(len(stars), dtype=np.float32))
        star_ids: list[int | None] = [None] * len(stars)
        for row in tracker.update(boxes):
            # Rows are x1, y1, x2, y2, track id, score, class (here the record index), detection index.
            star_ids[int(row[6])] = int(row[4])
        tracked, untracked = _split(registry, frame, stars, star_ids)
        registry.prune(frame)
        record["stars"], record["untracked_stars"] = tracked, untracked
        yield record


def _track_records_hungarian(records: Iterable[dict], params: HungarianParams) -> Iterator[dict]:
    """Star IDs from :class:`HungarianStarTracker`; an arm tracker lives as long as its star's track."""
    tracker = HungarianStarTracker(params)
    registry = ArmTrackers()
    provenance = star_tracking_key("hungarian", params)
    for frame, record in _frames_in_order(records):
        stars = record.get("stars") or []
        star_ids = tracker.update(stars, _record_time(record, frame))
        for ended in tracker.ended:
            registry.discard(ended)
        tracked, untracked = _split(registry, frame, stars, star_ids)
        record["stars"], record["untracked_stars"] = tracked, untracked
        record["star_tracking"] = provenance
        yield record


def star_tracking_key(star_tracker: str, params: HungarianParams) -> dict:
    """The ``star_tracking`` value ``track-videos`` appends to each record it tracks with the Hungarian tracker."""
    from . import __version__

    return {"tracker": star_tracker, "params": params.as_dict(), "version": __version__}


def resolve_star_tracker(star_tracker: str = "hungarian", star_params=None, tracker_config: Path | None = None):
    """``(star_tracker, HungarianParams or None, BoT-SORT config or None)``, checked."""
    if star_tracker not in STAR_TRACKERS:
        raise ValueError(f"star_tracker must be one of {STAR_TRACKERS}, got {star_tracker!r}")
    if star_tracker == "botsort":
        if star_params:
            raise ValueError("star-tracker parameters apply to the hungarian star tracker only")
        return star_tracker, None, Path(tracker_config) if tracker_config is not None else DEFAULT_BOTSORT_CONFIG
    if tracker_config is not None:
        raise ValueError("a BoT-SORT tracker config applies to --star-tracker botsort only")
    params = star_params if isinstance(star_params, HungarianParams) else HungarianParams.from_mapping(star_params)
    return star_tracker, params, None


def track_records(records: Iterable[dict], *, star_tracker: str = "hungarian", star_params=None,
                  tracker_config: Path | None = None) -> Iterator[dict]:
    """One video's records in frame order, with ``star_track_id`` set on stars and ``track_id`` on their arms.

    ``star_tracker="hungarian"`` (default) assigns star IDs with :class:`star_tracking.HungarianStarTracker`
    (``star_params``: a :class:`HungarianParams` or a mapping of its fields): every detection gets an ID in the
    record it appears in, except the less confident of two duplicate boxes, which moves to ``untracked_stars``. Each
    record gains a last key, ``star_tracking``, naming the tracker, its parameters and the package version.

    ``star_tracker="botsort"`` is the 0.5.0.dev12 behaviour, byte for byte: BoT-SORT as live replay runs it
    (``tracker_config``), whose unconfirmed detections (each star's first record after the video's first) move to
    ``untracked_stars``; records gain no ``star_tracking`` key.

    Either way nothing is dropped: ``stars`` plus ``untracked_stars`` are the record's detections. Arm IDs are scoped
    to their star (one released arm tracker per star ID), and ``arm_index`` is kept.
    """
    star_tracker, params, tracker_config = resolve_star_tracker(star_tracker, star_params, tracker_config)
    if star_tracker == "botsort":
        return _track_records_botsort(records, tracker_config)
    return _track_records_hungarian(records, params)


def read_records(path: Path) -> Iterator[dict]:
    if path.name.endswith(".zst"):
        with subprocess.Popen(["zstd", "-dcq", str(path)], stdout=subprocess.PIPE) as process:
            for line in process.stdout:
                yield json.loads(line)
        if process.returncode:
            raise OSError(f"zstd could not read {path}")
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def track_file(source: Path, destination: Path, tracker_config: Path | None = None, *,
               star_tracker: str = "hungarian", star_params=None) -> dict:
    """Track one video's records into ``destination``, compressed with zstd if it ends in ``.zst``.

    Returns the video's summary: counts, the star tracker and its parameters (or BoT-SORT config).
    """
    star_tracker, params, tracker_config = resolve_star_tracker(star_tracker, star_params, tracker_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    counts = {"video": str(source), "output": str(destination), "frames": 0, "tracked_stars": 0,
              "untracked_stars": 0, "star_ids": set(), "tracked_arms": 0, "star_tracker": star_tracker}
    if params is not None:
        counts["star_params"] = params.as_dict()
    else:
        counts["tracker_config"] = str(tracker_config)

    def lines():
        for record in track_records(read_records(source), star_tracker=star_tracker, star_params=params,
                                    tracker_config=tracker_config):
            counts["frames"] += 1
            counts["untracked_stars"] += len(record["untracked_stars"])
            for star in record["stars"]:
                counts["tracked_stars"] += 1
                counts["tracked_arms"] += len(star["arms"])
                counts["star_ids"].add(star["star_track_id"])
            yield (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()

    if destination.name.endswith(".zst"):
        with subprocess.Popen(["zstd", "-q", "-f", "-o", str(partial)], stdin=subprocess.PIPE) as process:
            for line in lines():
                process.stdin.write(line)
            process.stdin.close()
        if process.returncode:
            raise OSError(f"zstd could not write {partial}")
    else:
        with partial.open("wb") as stream:
            for line in lines():
                stream.write(line)
    os.replace(partial, destination)
    counts["star_ids"] = len(counts["star_ids"])
    return counts


def find_inputs(inputs: list[Path], output: Path) -> list[tuple[Path, Path]]:
    """``(source, destination)`` pairs; a directory is searched recursively and mirrored under ``output``."""
    pairs = []
    for item in inputs:
        if item.is_dir():
            sources = sorted(path for path in item.rglob("*") if path.is_file() and path.name.endswith(SUFFIXES)
                             and not path.name.startswith(("_", "worker.", "report.", "service.")))
            pairs += [(path, output / path.relative_to(item)) for path in sources]
        elif item.is_file() and item.name.endswith(SUFFIXES):
            pairs.append((item, output / item.name))
        else:
            raise FileNotFoundError(f"{item} is neither a directory nor a .jsonl or .jsonl.zst file")
    return pairs


def _track(job) -> dict:
    source, destination, star_tracker, params, tracker_config = job
    return track_file(source, destination, tracker_config, star_tracker=star_tracker, star_params=params)


def _parse_star_params(items: list[str]) -> dict:
    values = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--star-param takes KEY=VALUE, got {item!r}")
        values[key.strip()] = value.strip()
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="deepstarnoodles-inference track-videos", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="record files (.jsonl or .jsonl.zst) or directories")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the tracked records; directories are mirrored under it")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="videos tracked at once")
    parser.add_argument("--star-tracker", choices=STAR_TRACKERS, default="hungarian",
                        help="hungarian (default): the post-hoc star tracker for 1-2 fps records; botsort: BoT-SORT "
                             "as live replay runs it, the 0.5.0.dev12 behaviour")
    parser.add_argument("--star-param", action="append", default=[], metavar="KEY=VALUE",
                        help="override one Hungarian star-tracker parameter (repeatable): gate, gate_per_s, gate_max, "
                             "size_weight, size_gate, size_hist, max_gap_s, dup_iou, min_conf_new")
    parser.add_argument("--tracker-config", type=Path, default=None,
                        help=f"BoT-SORT configuration for --star-tracker botsort (default {DEFAULT_BOTSORT_CONFIG.name})")
    parser.add_argument("--overwrite", action="store_true", help="redo videos whose tracked records already exist")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    try:
        star_tracker, params, tracker_config = resolve_star_tracker(
            args.star_tracker, _parse_star_params(args.star_param), args.tracker_config)
    except ValueError as exc:
        parser.error(str(exc))
    if tracker_config is not None and not tracker_config.is_file():
        parser.error(f"--tracker-config {tracker_config} does not exist")
    pairs = find_inputs(args.inputs, args.output)
    if any(source.resolve() == destination.resolve() for source, destination in pairs):
        parser.error("--output would overwrite the input records; choose another directory")
    todo = [(source, destination, star_tracker, params, tracker_config) for source, destination in pairs
            if args.overwrite or not destination.exists()]
    settings = {"star_tracker": star_tracker}
    if params is not None:
        settings["star_params"] = params.as_dict()
    else:
        settings["tracker_config"] = str(tracker_config)
    print(json.dumps({"videos": len(pairs), "already_tracked": len(pairs) - len(todo), **settings}), flush=True)
    failed = 0
    log = None
    if todo:
        from . import __version__

        args.output.mkdir(parents=True, exist_ok=True)
        log = (args.output / RUN_LOG).open("a", encoding="utf-8")
        version = {"version": __version__}
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [(job[0], pool.submit(_track, job)) for job in todo]
            for source, future in futures:
                try:
                    summary = future.result()
                    print(json.dumps(summary), flush=True)
                except Exception as exc:  # One bad video must not stop the rest.
                    failed += 1
                    summary = {"video": str(source), "error": f"{type(exc).__name__}: {exc}", **settings}
                    print(json.dumps(summary), file=sys.stderr, flush=True)
                log.write(json.dumps({**summary, **version}) + "\n")
                log.flush()
    finally:
        if log is not None:
            log.close()
    return 1 if failed else 0
