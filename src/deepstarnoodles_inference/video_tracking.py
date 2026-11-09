# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Star and arm identities for videos that were annotated frame by frame.

``serve-videos`` annotates every frame on its own, so its records number arms per frame (``arm_index``) and leave
``star_track_id`` empty. ``track-videos`` replays over the saved records what live replay does during inference:
BoT-SORT over each frame's star boxes, then one arm tracker per star, which measures arms from that star's saved body
origin (``body_origin_model``). It needs no GPU, model weights or video. BoT-SORT runs without its camera-motion
compensation, which needs the frames; with a fixed camera that changes little. Records from before 0.5.0.dev11 have
no body origin and can't be tracked.
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


def track_records(records: Iterable[dict], *, tracker_config: Path = DEFAULT_BOTSORT_CONFIG) -> Iterator[dict]:
    """One video's records in frame order, with ``star_track_id`` set on stars and ``track_id`` on their arms.

    Live replay crops only the stars BoT-SORT has confirmed, which leaves out every star in the first frame it
    appears in after the video's first frame. Those detections move to the record's ``untracked_stars`` with null IDs,
    so ``stars`` holds what live replay would have and nothing is lost. Arm IDs are scoped to their star, and
    ``arm_index`` is kept.
    """
    from ultralytics.trackers.basetrack import BaseTrack

    tracker = star_tracker(tracker_config)
    registry = ArmTrackers()
    # Live replay numbers each video's stars from 1 (crop.TrackedStarCropper); this runs one video at a time.
    BaseTrack.reset_id()
    previous = None
    for record in records:
        frame = int(record["frame"])
        if previous is not None and frame <= previous:
            raise ValueError(f"frames must increase within a video; frame {frame} came after {previous}")
        previous = frame
        stars = record.get("stars") or []
        boxes = _Boxes(np.asarray([star["crop"]["bbox_xyxy"] for star in stars], dtype=np.float32).reshape(-1, 4),
                       np.asarray([star["crop"]["confidence"] for star in stars], dtype=np.float32),
                       np.zeros(len(stars), dtype=np.float32))
        star_ids: list[int | None] = [None] * len(stars)
        for row in tracker.update(boxes):
            # Rows are x1, y1, x2, y2, track id, score, class, detection index.
            star_ids[int(row[-1])] = int(row[4])
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
        registry.prune(frame)
        record["stars"], record["untracked_stars"] = tracked, untracked
        yield record


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


def track_file(source: Path, destination: Path, tracker_config: Path = DEFAULT_BOTSORT_CONFIG) -> dict:
    """Track one video's records into ``destination``, compressed with zstd if it ends in ``.zst``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    counts = {"video": str(source), "frames": 0, "tracked_stars": 0, "untracked_stars": 0, "star_ids": set(),
              "tracked_arms": 0}

    def lines():
        for record in track_records(read_records(source), tracker_config=tracker_config):
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
    return track_file(*job)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="deepstarnoodles-inference track-videos", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="record files (.jsonl or .jsonl.zst) or directories")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the tracked records; directories are mirrored under it")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="videos tracked at once")
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_BOTSORT_CONFIG)
    parser.add_argument("--overwrite", action="store_true", help="redo videos whose tracked records already exist")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    pairs = find_inputs(args.inputs, args.output)
    if any(source.resolve() == destination.resolve() for source, destination in pairs):
        parser.error("--output would overwrite the input records; choose another directory")
    todo = [(source, destination, args.tracker_config) for source, destination in pairs
            if args.overwrite or not destination.exists()]
    print(json.dumps({"videos": len(pairs), "already_tracked": len(pairs) - len(todo)}), flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [(job[0], pool.submit(_track, job)) for job in todo]
        for source, future in futures:
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:  # One bad video must not stop the rest.
                failed += 1
                print(json.dumps({"video": str(source), "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr,
                      flush=True)
    return 1 if failed else 0
