# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import itertools

import pytest

from deepstarnoodles_inference.star_tracking import HungarianParams, HungarianStarTracker

WIDTH = 200.0


def box(cx, cy, *, width=WIDTH, confidence=0.9):
    """A star whose box is ``width`` square around (cx, cy) in source px."""
    half = width / 2
    return {"crop": {"bbox_xyxy": [cx - half, cy - half, cx + half, cy + half], "confidence": confidence}}


def run(frames, params=None):
    """IDs per frame for ``[(time_s, [star, ...]), ...]``."""
    tracker = HungarianStarTracker(params)
    return [tracker.update(stars, time_s) for time_s, stars in frames], tracker


def test_a_three_width_jump_two_seconds_later_keeps_its_id():
    ids, _ = run([(0.0, [box(1000, 1000)]), (2.0, [box(1000 + 3 * WIDTH, 1000)])])
    assert ids == [[1], [1]]


def test_a_jump_beyond_gate_max_starts_a_new_id():
    # dt 10 s would allow 2 + 10 widths, but the gate is capped at 4.
    ids, _ = run([(0.0, [box(1000, 1000)]), (10.0, [box(1000 + 4.5 * WIDTH, 1000)])])
    assert ids == [[1], [2]]


def test_the_gate_grows_with_the_time_since_a_track_was_seen():
    ids, _ = run([(0.0, [box(1000, 1000)]), (0.5, [box(1000 + 2.7 * WIDTH, 1000)])])
    assert ids == [[1], [2]]          # 2.7 widths in 0.5 s: over 2 + 0.5
    ids, _ = run([(0.0, [box(1000, 1000)]), (1.0, [box(1000 + 2.7 * WIDTH, 1000)])])
    assert ids == [[1], [1]]          # 2.7 widths in 1 s: within 2 + 1


def test_a_track_unseen_longer_than_max_gap_s_ends():
    ids, tracker = run([(0.0, [box(1000, 1000)]), (30.0, [box(1000, 1000)])])
    assert ids == [[1], [1]] and tracker.ended == []
    tracker.update([], 60.0)
    assert tracker.ended == [] and tracker.live_ids == [1]   # exactly max_gap_s: still open
    assert tracker.update([box(1000, 1000)], 90.5) == [2]
    assert tracker.ended == [1] and tracker.live_ids == [2]
    params = HungarianParams(max_gap_s=120.0)
    ids, _ = run([(0.0, [box(1000, 1000)]), (100.0, [box(1000, 1000)])], params)
    assert ids == [[1], [1]]


def test_the_less_confident_of_two_duplicate_boxes_gets_no_id():
    first, twin = box(1000, 1000, confidence=0.6), box(1001, 1000, confidence=0.9)
    ids, _ = run([(0.0, [first, twin]), (0.5, [box(1002, 1000)])])
    assert ids == [[None, 1], [1]]
    ids, _ = run([(0.0, [first, twin])], HungarianParams(dup_iou=0.0))    # 0 turns it off
    assert ids == [[1, 2]]
    near = box(1000 + 0.2 * WIDTH, 1000)                                  # IoU ~0.67: two stars
    ids, _ = run([(0.0, [box(1000, 1000), near])])
    assert ids == [[1, 2]]


def test_a_size_change_over_size_gate_is_refused():
    ids, _ = run([(0.0, [box(1000, 1000)]), (0.5, [box(1000, 1000, width=2.1 * WIDTH)])])
    assert ids == [[1], [2]]
    ids, _ = run([(0.0, [box(1000, 1000)]), (0.5, [box(1000, 1000, width=1.9 * WIDTH)])])
    assert ids == [[1], [1]]
    ids, _ = run([(0.0, [box(1000, 1000)]), (0.5, [box(1000, 1000, width=WIDTH / 2.1)])])
    assert ids == [[1], [2]]


def test_two_stars_passing_each_other_keep_their_ids():
    frames = []
    for step in range(20):
        offset = 0.3 * WIDTH * step
        frames.append((float(step), [box(1000 + offset, 1000), box(1000 + 5.7 * WIDTH - offset, 1000 + WIDTH)]))
    ids, _ = run(frames)
    assert all(frame == [1, 2] for frame in ids)


def test_ids_are_numbered_from_one_in_order_of_birth():
    frames = [(0.0, [box(3000, 1000)]), (1.0, [box(3000, 1000), box(500, 500)]),
              (2.0, [box(8000, 8000), box(3000, 1000), box(500, 500)])]
    ids, _ = run(frames)
    assert ids == [[1], [1, 2], [3, 1, 2]]


def test_the_order_of_detections_in_a_record_does_not_change_the_partition():
    rows = [[box(1000 + 40 * t, 1000), box(2000, 1500 - 30 * t), box(400, 3000 + 25 * t, confidence=0.5)]
            for t in range(8)]
    reference = None
    for order in itertools.permutations(range(3)):
        tracker = HungarianStarTracker()
        groups: dict = {}
        for t, stars in enumerate(rows):
            for k, star_id in zip(order, tracker.update([stars[k] for k in order], float(t))):
                groups.setdefault(star_id, set()).add((t, k))
        partition = sorted(sorted(group) for group in groups.values())
        reference = reference or partition
        assert partition == reference
    assert len(reference) == 3


def test_min_conf_new_keeps_low_confidence_detections_from_starting_tracks():
    params = HungarianParams(min_conf_new=0.5)
    ids, _ = run([(0.0, [box(1000, 1000, confidence=0.3)]), (1.0, [box(1000, 1000, confidence=0.6)]),
                  (2.0, [box(1000, 1000, confidence=0.3)])], params)
    assert ids == [[None], [1], [1]]


def test_times_must_not_decrease():
    tracker = HungarianStarTracker()
    tracker.update([box(1000, 1000)], 5.0)
    with pytest.raises(ValueError, match="must not decrease"):
        tracker.update([box(1000, 1000)], 4.0)


def test_parameters_are_checked():
    assert HungarianParams.from_mapping({"gate": "2.5", "size_hist": "5"}) == HungarianParams(gate=2.5, size_hist=5)
    assert HungarianParams.from_mapping(None) == HungarianParams()
    with pytest.raises(ValueError, match="unknown star-tracker parameters: tau_s"):
        HungarianParams.from_mapping({"tau_s": 1})
    for bad in ({"gate": "5"}, {"gate": 0}, {"size_gate": 1.0}, {"max_gap_s": 0}, {"dup_iou": 1.5},
                {"size_hist": "2.5"}, {"size_hist": 0}, {"gate": "nan"}, {"gate_per_s": -1}, {"gate": "wide"},
                {"min_conf_new": 2}):
        with pytest.raises(ValueError):
            HungarianParams.from_mapping(bad)
    assert HungarianParams().as_dict() == {"gate": 2.0, "gate_per_s": 1.0, "gate_max": 4.0, "size_weight": 1.0,
                                           "size_gate": 2.0, "size_hist": 3, "max_gap_s": 60.0, "dup_iou": 0.95,
                                           "min_conf_new": 0.0}
