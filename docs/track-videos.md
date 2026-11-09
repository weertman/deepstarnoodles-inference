# Tracking after annotation

`deepstarnoodles-inference track-videos` gives the stars and arms in `serve-videos` records IDs that persist through a video. It needs no GPU, model weights or video, and runs one video per process:

```bash
deepstarnoodles-inference track-videos annotations/ --output tracked/ --jobs 8
deepstarnoodles-inference track-videos annotations/ --output tracked/ --star-param max_gap_s=120
deepstarnoodles-inference track-videos annotations/ --output tracked_dev12/ --star-tracker botsort
```

## What it does

1. **Star IDs.** By default (0.5.0.dev13 on) the post-hoc Hungarian star tracker (`star_tracking.py`) assigns them. `--star-tracker botsort` gives the 0.5.0.dev12 IDs instead (see [BoT-SORT](#bot-sort-the-050dev12-behaviour)).
2. **Arm IDs.** One arm tracker per star ID follows its arms, measured from the star's saved `body_origin_model`. It is the released FusedTracker (shape gate 120 px, embedding weight 2.0, max age 16 arm-bearing updates, match-cost cap 90, 12-point tip-first resample), run through the vendored `noodle_polylines` array tracker. It gives the same arm IDs as live replay's `runtime.StarArmTrackerRegistry` bit for bit, at a fraction of the CPU. Arm IDs are scoped to their star. A record where a star has no arms does not age that star's arm tracks.

Each star gets `star_track_id`, and each arm gets `track_id`; `arm_index` stays. A detection without a star ID moves to the record's `untracked_stars`, with null IDs, so `stars` plus `untracked_stars` are always exactly the record's detections. The output reads in `representations` without `--untracked`.

## The Hungarian star tracker (default)

At 1–2 fps a moving star travels about a body width between records. BoT-SORT matches boxes on overlap (IoU) and waits a frame or two before confirming a new track. At these rates it therefore splits one star into many IDs and holds each new track's first detection back. On the two 4-minute, 2 fps flipping clips of the backcast rehearsal, one star got 7 and 11 IDs. The Hungarian tracker was built for these records.

For each record it solves one assignment (`scipy.optimize.linear_sum_assignment`) between the open tracks and the record's detections. The cost of a pair is

    cost = d + size_weight * |ln(s_det / s_track)|

where `s = sqrt(box area)`, and `d` is the distance between the box centres divided by `sqrt(s_track * s_det)`. A track's position is its last box centre. It has no motion model, because a velocity model made the flip clips worse. Its size is the median of its last `size_hist` sizes. A pair is allowed only when `d <= min(gate + gate_per_s * dt, gate_max)` and the sizes differ by at most a factor `size_gate`. Here `dt` is the seconds since the track was last seen, taken from the records' `time_s`, or `frame / fps`, or the frame number when neither is present.

- **New IDs.** Every detection left unmatched starts a new ID in the record it appears in. IDs are numbered from 1 in order of birth.
- **Duplicate boxes.** When two boxes in one record have IoU >= `dup_iou`, the less confident one gets no ID and goes to `untracked_stars`.
- **Gaps.** A track unseen for more than `max_gap_s` seconds ends, and its arm tracker is dropped. Gaps shorter than that, including missing records, keep the star's ID and its arm IDs; the gate grows with the gap.
- **Determinism.** The tracker has no randomness. The order of detections within a record does not change which detections share an ID.

| Parameter | Default | Meaning |
|---|---|---|
| `gate` | 2.0 | max normalised centre distance at `dt` → 0 |
| `gate_per_s` | 1.0 | extra normalised distance allowed per second since the track was seen |
| `gate_max` | 4.0 | cap on the gate |
| `size_weight` | 1.0 | weight of the log size ratio in the cost |
| `size_gate` | 2.0 | max size ratio, either way |
| `size_hist` | 3 | a track's size is the median of its last N sizes |
| `max_gap_s` | 60.0 | a track unseen for longer ends |
| `dup_iou` | 0.95 | same-record IoU at which the less confident box is a duplicate; 0 turns it off |
| `min_conf_new` | 0.0 | detections below this confidence start no track |

`--star-param KEY=VALUE` overrides one parameter and may be repeated. Values are checked before any video starts: `0 < gate <= gate_max`, `size_gate > 1`, `max_gap_s > 0`, `0 <= dup_iou <= 1`, a positive integer `size_hist`, and unknown keys are refused. The defaults gave the same flip-clip result for `gate` from 2.0 to 3.0 and `max_gap_s` from 30 to 300.

### Known costs

- **Hand-overs merge.** With no appearance cue, the tracker cannot tell one animal leaving from another arriving nearby. In 9 synthetic hand-over cases (two animals never in view together), it gave both one ID every time; tuned BoT-SORT did in 6. This is the price of bridging 3-width jumps and 15 s gaps.
- **IDs carry across cuts.** The synthetic lab clips cut between unrelated stills. An ID carried to a star whose box clearly moved 53 times in one clip and 33 in the other, against 17 and 13 for BoT-SORT. Real footage has no cuts, but a star swapped for another in the same place between records would keep the ID.
- **One-frame false positives get IDs.** BoT-SORT's confirmation delay used to hide them in `untracked_stars`. They are now short tracks of their own.
- **Arm-ID churn inside a long star track becomes visible.** On one flip clip at 1 fps, arm ID switches rose from 28 to 56 per 1,000 links. That comes from the arm tracker, which now sees the whole star rather than fragments.
- **Tested on two single-star clips and synthetic stills only.** The gates have not been checked on real multi-star 1 fps motion. Check the default against a hand-checked real 1 fps lab segment before a large run.

## BoT-SORT (the 0.5.0.dev12 behaviour)

`--star-tracker botsort` runs BoT-SORT with the configuration live replay uses (`--tracker-config FILE` to change it). It has no camera-motion compensation, which needs the frames. Its output is byte-identical to 0.5.0.dev12 on current data; the tests pin it on two real fixtures. Stars BoT-SORT has not confirmed, which includes each star in the first record it appears in after the video's first, go to `untracked_stars`, as live replay never crops them. A star unseen for more than 30 frames loses its arm tracker. `--star-param` with `botsort`, and `--tracker-config` with `hungarian`, are refused.

One fix applies to this path. In ultralytics 8.4.51, a BoT-SORT row's detection index counts within the high- or low-confidence subset it came from, not within the record. `track-videos` now carries each box's record index through BoT-SORT's unused class column. With the shipped configuration (`fuse_score: true`) and `serve-videos`' 0.25 confidence floor no low-confidence box is ever matched, so the output is unchanged. Live replay's own BoT-SORT path (`crop.py`) is not changed.

## Recording what was used

- **Hungarian output.** Each tracked record gets a last key, `star_tracking`, of the form `{"tracker": "hungarian", "params": {...}, "version": "0.5.0.dev13"}`. Keys are only appended, as for the output schemas.
- **BoT-SORT output.** It gets no such key, to stay byte-identical to dev12. A tracked record that has `untracked_stars` but no `star_tracking` came from BoT-SORT, from dev12 or from `--star-tracker botsort`.
- **Summaries.** Each video's summary line on stdout gives its counts, the tracker and its parameters or BoT-SORT config. The line is also appended, with the package version, to `_track-videos.jsonl` in the output directory. Files starting with `_` are never read as inputs by either command.

## Guarantees and refusals

- **Run once, on annotations.** Records that already have star IDs (`untracked_stars`, `star_tracking`, or a non-null `star_track_id`, as `track-videos`, `live` and `live-replay` write them) are refused, not tracked again. Records without a `stars` list are refused. Records from before 0.5.0.dev11 have no body origin and are refused.
- **Resuming and idempotence.** A video whose output exists is skipped unless `--overwrite` is given. Tracking is deterministic, so a rerun with `--overwrite` writes the same bytes.
- **Layout.** Directories are searched recursively and mirrored under `--output`. So `cam1/C0001.mp4.jsonl.zst` and `cam2/C0001.mp4.jsonl.zst` stay apart. The fleet's `_manifest.jsonl` and `worker.*`, `report.*` and `service.*` files are skipped. An `--output` inside an input directory is not read back on the next run. An `--output` equal to an input directory is refused. So are one video in two formats (`x.jsonl` next to `x.jsonl.zst`) and two inputs that would land on the same output.
- **Bad files.** A file without records is an error. A line that is not JSON, such as a truncated last line, is reported with the file, line number and last good frame. A truncated `.jsonl.zst` reports `zstd`'s own error. Blank lines are skipped. A failed video leaves no `.part` file, and the run goes on with the next video and exits 1 at the end.
- **Compressed files.** `.jsonl.zst` is read and written through `zstd` on `PATH`. Input goes to it on stdin, so symlinked files work.
