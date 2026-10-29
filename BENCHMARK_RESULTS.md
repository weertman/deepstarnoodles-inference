# Inference benchmark results — completed pilot through Round 4, 28–29 October 2026 (UTC)

**For archives, use singleton detector/crop batches, minimal Hydra outputs, fixed GRU with check interval 1, and two bounded decode workers.** The final same-build comparison improved zero-warmup whole-process throughput by **32.72% and 22.23%** in two paired runs (**27.48% ratio-of-means gain**); process wall fell **24.65% and 18.19%**. A separate comparison held decode workers fixed and measured the added minimal/fixed profile at **+6.76% / +7.18% processing throughput**. Strict output parity passed throughout.

**For live use, singleton minimal/fixed1 is a qualified warmed-replay candidate with synchronous decoding.** Both repeats improved p95 on every positive clip, including the sparse crowded probe. The detector-only negative clip varied sharply and regressed in one repeat. This supports testing the candidate on the target machine; it does not establish a uniform latency reduction, a camera FPS guarantee, or sustained queue behavior. Earlier zero-warmup live results remain variable and are retained below.

These experiments ran on an **RTX 5090**, with resource guards intended to leave headroom for an 8 GiB GPU and 32 GiB host. They do **not** establish RTX 5060 performance or certify physical 8 GiB fit. The trained cascade is **YOLO26-L segmentation → DINOv3 ConvNeXt-Base Hydra → GRU128**; the Hydra checkpoint is Base, not Large.

**All four rounds are complete:** the pilot, Round 2, Round 3 and Round 4 contain **43, 112, 80 and 30 recovered, hash-verified cases**. Round 4 has no deferred cases, 24 strict parity passes and six uncompared references; both archive blocks and all five warmed live blocks qualified. Its final evidence was verified on 2026-10-29 at 01:52:57 UTC.

## Round 4: combined archive configuration and warmed live verification

The source and container are identical to Round 3. Detector/crop batch stays **1**; default Hydra batching therefore executes one crop at a time. Weights, precision/layout settings, thresholds, seed rules and tolerances are unchanged. Decode prefetch retains at most four frames within a 256 MiB decoded-array budget, with a 64 MiB maximum frame. Cross-frame crop buffering and staged execution remain disabled.

### Added profile: decode workers held fixed at two

C uses full Hydra outputs and dynamic GRU; D uses minimal outputs and fixed GRU with check interval 1. Each warms four frames (eight crops and 78 arms), then measures the same 256-image workload. Chronological order is C1, D1, D2, C2.

| Order | Profile | Processing images/s | Process wall, s | Benchmark job, s |
|---|---|---:|---:|---:|
| C1 | Full / dynamic | 4.431 | 67.909 | 63.745 |
| D1 | Minimal / fixed1 | 4.730 | 64.197 | 60.416 |
| D2 | Minimal / fixed1 | 4.761 | 63.708 | 59.945 |
| C2 | Full / dynamic | 4.442 | 66.701 | 62.944 |

The added profile improves processing throughput **6.97%** by ratio of means, paired **+6.76% / +7.18%**. Candidate rates span **4.730–4.761 images/s**, versus **4.431–4.442** for the control. Process wall improves **4.98%** and benchmark-job time **4.99%** by ratio of means. This isolates the added profile while prefetch stays enabled; it does not measure prefetch alone.

### Combined archive configuration: zero-warmup whole jobs

A is full/dynamic with sequential decoding. D combines minimal/fixed1 with two decode workers. All runs measure the same 256 inputs with zero warmup, ordered A1, D1, D2, A2.

| Order | Configuration | Process wall, s | Whole-process images/s | Benchmark job, s |
|---|---|---:|---:|---:|
| A1 | Eager / decode0 | 83.921 | 3.050 | 80.001 |
| D1 | Combined / decode2 | 63.230 | 4.049 | 59.267 |
| D2 | Combined / decode2 | 68.669 | 3.728 | 64.731 |
| A2 | Eager / decode0 | 83.935 | 3.050 | 80.052 |

Whole-process throughput improves **27.48%** by ratio of means, paired **+32.72% / +22.23%**. Process wall changes **−21.42%**, paired **−24.65% / −18.19%**; benchmark-job time changes **−22.53%**, paired **−25.92% / −19.14%**. Combined rates span **3.728–4.049 whole-process images/s**, versus approximately **3.050** in both eager repeats. Two-repeat ranges are descriptive, not confidence intervals.

Every archive run produces **454 crops and 4,554 arms** and passes unchanged gates when compared with the reference. Outputs repeat byte-identically within each configuration; fixed versus dynamic outputs differ within tolerance. The eager output matches prior rounds byte for byte. Each archive repeat peaks at **4,004 MiB sampled device memory**; the largest tree RSS across these blocks is **3.12 GiB**. The initial unpaired no-prefetch singleton validation was slightly slower than the first eager observation (3.484 versus 3.576 images/s); the recommendation rests on the repeated comparisons above.

### Warmed live replay: positive clips improve, negative control varies

Both A (full/dynamic) and B (minimal/fixed1) warm the **entire clip**, then reset tracking and replay it: 16 frames per ROV clip and three for the sparse crowded probe. Decode stays synchronous and detector/crop batch stays 1. Each clip has its own A1, B1, B2, A2 block. Negative percentages mean lower per-run p95.

| Clip | Reference p95 A1 / A2, ms | Candidate p95 B1 / B2, ms | Mean-ratio change | Paired changes |
|---|---:|---:|---:|---:|
| ROV A | 204.7 / 192.5 | 173.4 / 154.9 | -17.35% | -15.28% / -19.56% |
| ROV B | 174.2 / 200.0 | 156.9 / 152.2 | -17.42% | -9.94% / -23.94% |
| ROV C | 147.9 / 178.7 | 135.7 / 144.3 | -14.28% | -8.23% / -19.28% |
| Negative ROV | 23.9 / 22.6 | 22.7 / 38.6 | +31.71% | -5.21% / +70.79% |
| Sparse crowded | 620.8 / 702.6 | 581.2 / 648.2 | -7.11% | -6.37% / -7.75% |

All five blocks pass strict parity. Positive ROV clips detect **14, 10 and three crops** (237, 45 and 49 arms); the crowded probe detects **6 / 4 / 4 animals** in its three frames (85 arms). Warmup exercises the same detections and arms as the timed pass. Each configuration repeats byte-identically. The negative clip has no detections and calls neither Hydra nor GRU; its **−5.21% / +70.79%** paired changes demonstrate timing variation and cannot be credited to their altered execution.

The positive clips support minimal/fixed1 as the next live configuration to validate, while the negative result prevents a uniform speedup claim. These are warm-cache, short, same-shape replays on a 5090, with only three crowded frames. They do not measure sustained camera input, queue backlog, dropped frames, or a stable long-run tail. Earlier-round differences cannot isolate warmup as a cause because host and profile also differ.

## Round 3: decode prefetch with unchanged model execution

A uses sequential decoding; B uses **two decode workers**, four prefetched frames, a 256 MiB retained decoded-array budget and a 64 MiB maximum frame. Both keep detector/crop batch 1, full Hydra outputs, dynamic GRU, default Hydra batching and unchanged precision/layout. No crop buffer is enabled. Execution order is A1, B1, B2, A2; all runs use the same 256 inputs.

### Warmed processing: four warmup frames

| Order | Decode workers | Processing images/s | Process wall, s | Benchmark job, s |
|---|---:|---:|---:|---:|
| A1 | 0 | 3.229 | 89.306 | 85.731 |
| B1 | 2 | 4.520 | 64.578 | 61.029 |
| B2 | 2 | 5.556 | 53.567 | 50.148 |
| A2 | 0 | 3.679 | 77.086 | 73.698 |

Processing throughput’s B/A ratio of means is **1.4586 (+45.86%)**; paired ratios are **1.4000 / 1.5101**. Reference repeats span **3.229–3.679 images/s** and candidate repeats **4.520–5.556**. This variation is material: two-repeat ranges are descriptive, not confidence intervals. Process-wall and benchmark-job ratios of means are 0.7100 (−29.00%) and 0.6973 (−30.27%), but these timings include four warmup images.

### Whole jobs: zero warmup

| Order | Decode workers | Process wall, s | Whole-process images/s | Benchmark job, s |
|---|---:|---:|---:|---:|
| A1 | 0 | 69.136 | 3.703 | 65.177 |
| B1 | 2 | 53.244 | 4.808 | 50.486 |
| B2 | 2 | 59.566 | 4.298 | 55.522 |
| A2 | 0 | 66.055 | 3.876 | 62.685 |

Whole-process throughput’s B/A ratio of means is **1.2016 (+20.16%)**, with paired gains **+29.85% / +10.89%**. Process wall is **−16.56%** by ratio of means, with paired changes **−22.99% / −9.82%**; benchmark-job time is **−17.09%**, paired **−22.54% / −11.43%**. Both pairs favor prefetch, while the spread limits precision. The warmed and zero-warmup effects should not be collapsed into one speedup percentage.

All eight archive repeat files are byte-identical to their same-build reference and contain 454 detected crops and 4,554 arms. Sampled device peaks stay at 4,004 MiB. The warmed block’s largest tree RSS is 3.14 GiB; the zero-warmup block’s is 3.10 GiB. Single full 256-frame promotions also passed with one and two workers, but only two workers received ABBA confirmation. This does not prove two workers are faster than one.

The prefetch reader preserves input order and bounds retained decoded arrays. In the full 256-frame promotion it peaked at four pending frames and 110.84 MiB reserved decoded arrays. Worker decode overlaps inference; worker elapsed time is not additive to consumer-stage timings or guaranteed time saved. Filesystem-cache and storage limits described below still apply.

### Live replay: parity passed, latency mixed

The Round 3 live candidate uses minimal Hydra outputs, Hydra batch 1, dynamic GRU and crop batch 1, with sequential decoding and zero warmup. A uses full outputs and default Hydra batching; the singleton crop keeps effective Hydra execution at one. This is distinct from Round 2’s fixed-GRU candidate. Each clip gets its own A1, B1, B2, A2 block and tracking reset. Negative percentage means lower p95 latency.

| Clip | Reference p95 A1 / A2, ms | Candidate p95 B1 / B2, ms | Mean-ratio change | Paired changes |
|---|---:|---:|---:|---:|
| ROV A | 569.4 / 485.7 | 559.3 / 380.5 | -10.93% | -1.78% / -21.65% |
| ROV B | 439.4 / 762.0 | 453.3 / 638.9 | -9.09% | +3.16% / -16.15% |
| ROV C | 506.6 / 538.4 | 409.5 / 578.0 | -5.50% | -19.16% / +7.35% |
| Negative ROV | 127.0 / 153.2 | 103.7 / 106.1 | -25.12% | -18.35% / -30.74% |
| Sparse crowded | 1308.8 / 1700.1 | 1936.0 / 1542.1 | +15.59% | +47.92% / -9.29% |

All five blocks passed the unchanged gates, and each block’s four outputs are byte-identical. ROV B and C each have one regression; the sparse crowded mean p95 is **15.59% slower**, with paired changes **+47.92% / −9.29%**. The negative clip invokes neither Hydra nor GRU, so its apparent improvement cannot be attributed to them. These are per-run short-replay p95 estimates, not pooled percentiles or sustained-camera latency. No broad live speedup is established.

### Screen rejections

Holding Hydra at batch 1 did not make crop/GRU batches 2/4/8 equivalent: the 64-frame dynamic screens still failed strict parity. Fixed GRU with check interval 1 at crop batch 2 and minimal outputs at crop batch 2 also failed. Buffered crop batches 8/16/32 and staged batches 8/16/32 each failed, so neither schedule was promoted. All 11 archive rejections recorded ten parity events apiece: confidence, length, point-count and steps; boxes and compared identities matched. Actual buffered groups reached the requested 8/16/32 sizes within their configured queue limits. Buffered ingestion p95 is not frame-completion latency and is not used for promotion claims.

A separate matched-seed diagnostic on the verified 64-frame outputs localizes the material difference to the **same two arms taking one extra step** across grouped dynamic crop batches 2/4/8 and the minimal-output/fixed-slot crop-2 variants. All 110 crops, 1,103 seed identities and tip scores match. This is consistent with batch-shape numerical sensitivity amplified by a discrete stopping decision, but intermediate logits were not captured and the first divergent kernel is not established. The strict gates remain unchanged and these grouped cases remain rejected. The numeric evidence records the diagnostic digest and its correspondence to the final recovered case hashes.

The crowded live probe failed at crop batches 2/4/8 even with Hydra batch 1, with 17 events each including arm tracking IDs. Singleton full/dynamic Hydra controls are baseline-equivalent and were excluded from optimization selection. Prefix screens and their fastest observations are not the report’s main conclusion.

Round 4 completed the same-build singleton combination and warmed-live follow-up; its repeated results and remaining limits appear above.

## Round 2: repeated resident measurements

The reference uses full Hydra outputs and dynamic GRU; the candidate uses minimal outputs, fixed GRU and check interval 1. Both use detector/crop batch 1 with identical weights and precision.

Each run processes the first four inputs once as warmup, then times all 256 inputs, including those same four frames. A is the reference and B the candidate; execution order is A1, B1, B2, A2.

| Execution order | Configuration | Processing images/s | Child-process seconds | Child-process images/s | Benchmark job seconds | Device peak MiB | RSS peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Reference | 3.584 | 80.318 | 3.187 | 76.838 | 4004 | 3.06 |
| B1 | Candidate | 3.733 | 77.044 | 3.323 | 73.978 | 4004 | 2.97 |
| B2 | Candidate | 3.759 | 76.591 | 3.342 | 73.345 | 4004 | 3.07 |
| A2 | Reference | 3.559 | 80.756 | 3.170 | 77.452 | 4004 | 3.05 |

The two within-pair processing-rate changes are **+4.16% / +5.61%**. Child-process rates changed **+4.25% / +5.44%**, with process time **−4.08% / −5.16%**. The candidate’s two processing rates span 3.733–3.759 images/s, versus 3.559–3.584 for the reference. These two repeats describe observed variation; they are not confidence intervals. The [numeric evidence](BENCHMARK_EVIDENCE.json) retains unrounded paired observations and verified-manifest hashes. Every repeat passed parity and produced 454 detected star crops and 4,554 arm records. Predictions were byte-identical within each configuration’s repeats; candidate/reference files differ but satisfy the numeric and structural gates.

Three singleton-crop configurations reached full-256 validation: minimal outputs with fixed GRU check intervals 1, 4 and 8. Their initial processing rates were 3.747, 3.621 and 3.614 images/s, respectively, and all passed. Only interval 1 received the repeated comparison above. The singleton reference remains the fallback where a candidate lacks a demonstrated benefit.

**Timing definitions matter.** Processing images/s covers the timed inference loop, excluding initialization and warmup. “Benchmark job seconds” runs from the benchmark’s main function to its final timing sample, including model initialization, warmup, inference and prediction output (and staged spool cleanup); earlier Python/module imports and final report serialization are excluded. “Child-process seconds” is the monitor’s launch-to-observed-exit time, including startup and polling overhead. Its rate is 256 divided by those seconds; for the table above, the wall time also includes four warmup frames. It is therefore not a zero-warmup job comparison.

## Round 2 and pilot: staged versus resident whole jobs

The next table uses the selected candidate with **zero warmup**, the same 256 frames, and detector/crop batch 1. A is resident execution; B stages 64-frame shards through the models. Both pairs passed parity.

| Execution order | Schedule | Child-process seconds | Whole-process images/s | Benchmark job seconds | Device peak MiB | RSS peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Resident | 76.006 | 3.368 | 72.998 | 4004 | 2.83 |
| B1 | Staged, 64 frames | 88.759 | 2.884 | 85.606 | 4228 | 3.40 |
| B2 | Staged, 64 frames | 86.564 | 2.957 | 83.313 | 4228 | 3.41 |
| A2 | Resident | 75.987 | 3.369 | 72.916 | 4004 | 2.98 |

Staging reduced whole-process throughput by **14.37% / 12.22%** and increased wall time by **16.78% / 13.92%**. It also had higher sampled device memory here. The resident schedule is the better measured throughput choice for this workload. Processing-only rates are intentionally omitted: staging’s model reload and spool costs make those rates a different comparison scope. Each schedule’s two output files were byte-identical; staged/resident outputs passed parity without being byte-identical. Optional Round 2 shard-32 and shard-128 runs also passed: they took 99.317 / 84.011 process seconds (2.578 / 3.047 images/s), with sampled device peaks 3,058 / 3,706 MiB and RSS peaks 3.44 / 3.43 GiB. Those sizes have one observation each, not paired confirmation.

The completed pilot also tested shard sizes 32/64/128, each once with the original reference and zero warmup:

| Pilot schedule | Process seconds | Whole-process images/s | Sampled device peak MiB | Sampled RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Resident | 80.179 | 3.193 | 4,004 | 3.00 |
| Staged, 32 frames | 99.675 | 2.568 | 3,058 | 3.43 |
| Staged, 64 frames | 93.590 | 2.735 | 4,228 | 3.40 |
| Staged, 128 frames | 88.454 | 2.894 | 3,706 | 3.39 |

All four passed parity. The 32-frame schedule traded 23.6% lower sampled device peak for 19.6% lower whole-process throughput. Each staged job wrote about 12.38 GiB cumulatively; peak live spool was 1.35/2.48/4.78 GiB for shards 32/64/128, then cleaned to zero. This is a memory/throughput tradeoff, not a throughput win. Do not compare pilot and Round 2 timings as a cross-build speedup.

## Round 2: repeated per-frame latency

Each clip starts a fresh process and tracking state, with zero warmup. The same singleton-crop candidate passed every frame of all five clips in both repeats. Pair 1 runs reference then candidate; pair 2 reverses the order. Rates below are measured processing rates; p95 is observed frame-completion latency within each short replay. Negative percentage means lower latency.

| Clip | Pair | Reference p95 ms | Candidate p95 ms | p95 change | Reference processing images/s | Candidate processing images/s |
| --- | --- | --- | --- | --- | --- | --- |
| ROV A | 1 | 501.7 | 491.6 | -2.01% | 5.245 | 5.675 |
| ROV A | 2 | 513.3 | 495.1 | -3.54% | 5.108 | 5.560 |
| ROV B | 1 | 508.8 | 482.6 | -5.14% | 5.906 | 6.387 |
| ROV B | 2 | 594.7 | 506.1 | -14.90% | 5.574 | 6.194 |
| ROV C | 1 | 494.1 | 502.6 | +1.71% | 8.964 | 9.506 |
| ROV C | 2 | 519.9 | 512.6 | -1.39% | 8.824 | 9.411 |
| Negative ROV | 1 | 169.6 | 141.3 | -16.67% | 16.845 | 19.099 |
| Negative ROV | 2 | 185.4 | 138.1 | -25.50% | 15.967 | 19.165 |
| Sparse crowded | 1 | 1789.1 | 1484.0 | -17.06% | 1.039 | 1.199 |
| Sparse crowded | 2 | 1576.2 | 1649.9 | +4.67% | 1.133 | 1.131 |

ROV A/B/C each contain 16 frames at 224×224, with 14/10/3 detected-positive frames and at most one detected star per frame. The negative ROV clip contains 16 frames and runs no Hydra/GRU inference; its apparent improvement cannot be attributed to those models. The sparse crowded clip contains only three 960×600 frames, with 6, 4 and 4 detected stars. It is not a continuous camera stream. Every configuration’s predictions were byte-identical across its two repeats.

ROV C and the crowded clip each have one latency regression. The crowded processing-rate change is +15.41% in pair 1 and −0.22% in pair 2. These tiny replays are useful for checking behavior and exposing regressions, but do not establish sustained FPS, queue behavior, dropped frames, capture-to-display latency, or a reliable latency tail. The live result remains workload-specific; retain the reference for unproven conditions. Earlier single-screen latency improvements are not used as the repeated conclusion.

## Why larger batches were not selected

In the completed pilot, crop batches 2/4/8 reached 4.005/3.913/4.342 warmed images/s but failed parity with 129/92/142 issue events. Their detector boxes and raster masks matched, while downstream arm counts, trace steps, confidence, identities or geometry changed. Faster execution with changed outputs does not qualify as an equivalent inference optimization. Issue counts are events, not failed-frame counts.

Round 2’s 64-frame archive screen had four passing singleton-crop variants, 20 parity-failing batched variants, and eight detector-batch-16 variants stopped by the sampled device-memory guard. The latter reached 7,568 MiB, above the 7 GiB target. Four crop-batch-4 live variants passed the single-star ROV clips but failed the crowded probe; those ROV passes never exercised a crop batch above one. No failed batch was promoted. The complete pilot likewise promoted none of its eight archive alternatives and no live alternative across all five probes.

The unchanged gate requires structural agreement, including counts, trace steps and stopping reasons; minimum mask IoU 0.99; box/polyline/length differences within 1 pixel in their specified coordinate spaces; identity/confidence differences within 0.01; and source-mapping difference within 0.01 pixel. This is runtime parity relative to the pinned eager reference, not a measure of biological accuracy.

## Workload and memory limits

The archive has 256 unique frames from 14 sources and 82 capture groups: 136 lab, 60 field and 60 web images, stratified across domain and crowding rather than weighted to production frequency. Annotations mark 177 positive and 79 negative frames, 426 animals and a maximum of 16 per frame; the reference actually detects 454 crops, at most nine in a frame. Native widths range from 224 to 5,496 and heights from 224 to 3,672. The archive and live workloads together have 291 unique images; 32 live images also occur in the archive.

All inputs come from StarSeg validation. Known Hydra test source/group matches were excluded, but the archive includes 93 Hydra training matches, one train-and-validation match, 19 validation matches and 143 unverified cases. An absent match does not prove independence. These data support runtime comparison, not accuracy or generalization claims.

Archive image files total 598.65 MiB and fit in host RAM. Hashing inputs before inference warms the filesystem cache. This is not a cold-disk, remote-storage or uncached large-archive benchmark. Output flushing does not include fsync, so the reported rate is not durable-storage throughput.

All 30 Round 4 cases have verified resource samples: **1,741 samples**, no sampling errors, missing device/RSS observations, guard exceedances or stop reasons. All-case sampled peaks are **4,344 MiB device and 3.115 GiB tree RSS**. The maximum sampling gap is **4.329 seconds**, in one archive profile candidate run; this is materially longer than earlier rounds and further limits peak-memory assurance. Sampling does not prove the absence of brief higher peaks.

All 80 Round 3 cases have verified resource samples: **2,783 samples**, no sampling errors, missing device/RSS observations, guard exceedances or stop reasons. The maximum observed gap is 0.659 s; all-case peaks are 4,344 MiB device and 3.934 GiB process-tree RSS. These are sampled peaks, not hard upper bounds.

All 112 recovered Round 2 cases have verified resource samples: 4,342 samples, no missing device observations or sampling-error entries, and eight guard stops. The maximum observed sample gap is 0.911 seconds. The selected archive repeats stayed at 4,004 MiB sampled device peak and about 2.97–3.07 GiB sampled process-tree RSS; accepted live repeats reached 4,344 MiB and about 2.62 GiB. The eight rejected large-detector batches raised the all-case device maximum to 7,568 MiB. All-case RSS peaked at 3.44 GiB. The completed pilot had 3,159 samples, no guard stops, a 4,602 MiB device maximum and 3.43 GiB RSS maximum.

The campaign combines a 7 GiB PyTorch allocator limit with sampled 7 GiB whole-device and 24 GiB process-tree RSS guards. Sampling can miss brief peaks, and an allocator cap does not constrain all GPU allocations. These guards leave the RTX 5090’s compute, bandwidth and physical memory available; they are neither an RTX 5060 emulator nor a hard 32 GiB host-memory limit. Verification on the actual 8 GiB target and sustained representative live inputs is still required.

## Reproduce the final Round 4 configuration

Use the immutable Round 4 image in the build table and the same model manifest, image bytes and input order for each comparison. Set OMP_NUM_THREADS=8, MKL_NUM_THREADS=8 and OPENBLAS_NUM_THREADS=1. Choose a fresh output directory each time.

```bash
# Combined archive candidate: zero-warmup whole job.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round4-archive --images 256 --warmup 0 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Warmed live replay candidate; use --warmup 3 for a three-frame probe.
deepstarnoodles-inference live-replay \
  --models /models/manifest.json --index /inputs/clip.csv \
  --output /outputs/round4-warm-live --warmup 16 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --decode-workers 0 --allocator-limit-gib 7 --ram-limit-gib 24
```

For the archive whole-job reference, use full Hydra outputs, dynamic GRU, check interval 8 and decode workers 0. To isolate the added profile, keep decode workers 2 on both sides and use warmup 4. For the live reference, change only the three profile flags to full/dynamic/8; keep full-clip warmup and synchronous decode. Default Hydra batching is unchanged. Use the public compare command against a same-build reference before adopting a candidate. Private benchmark images are not distributed; use representative local inputs for target-machine validation.

## Reproduce Round 3 and Round 2

For the Round 3 archive result, use the immutable Round 3 image in the build table. Keep the models, input CSV and mounted images identical between A and B. Match `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=1`, and choose a new output directory each time.

```bash
# Round 3 B: warmed processing; use --warmup 0 for the whole-job comparison.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round3-decode2 --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode full --gru-execution dynamic --gru-check-interval 8 \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24
```

For Round 3 A, change only `--decode-workers 2` to `--decode-workers 0` and choose a fresh output directory. Its live comparison uses zero-warmup `live-replay`, sequential decode, crop batch 1, `--hydra-batch-size 1 --hydra-output-mode minimal --gru-execution dynamic --gru-check-interval 8`; it is a measured parity-valid configuration, not a general latency recommendation.

Use the immutable Round 2 image below and the pinned model manifest. These commands assume its CLI is available in the container, with models and CSV/image inputs mounted read-only at `/models` and `/inputs`, and a writable `/outputs`. Set `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=1` to match the worker. Choose an unused output-directory path for each execution; the CLI creates it and rejects existing directories. The private workload is not distributed with this report, so substitute your own representative CSV when it is unavailable.

```bash
# Archive candidate, zero-warmup complete-job configuration.
deepstarnoodles-inference archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/archive-candidate --warmup 0 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Archive repeated-processing configuration: same flags, 256 frames + 4 warmup.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/archive-warmed --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Live replay: chronological frames; run each clip separately to reset tracking.
deepstarnoodles-inference live-replay \
  --models /models/manifest.json --index /inputs/clip.csv \
  --output /outputs/live-replay --warmup 0 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --allocator-limit-gib 7 --ram-limit-gib 24
```

For the corresponding reference, replace the three candidate flags with `--hydra-output-mode full --gru-execution dynamic --gru-check-interval 8`. Archive mode treats frames independently; live replay maintains tracking state within each clip. The `live` camera command is a separate path and was not benchmarked here. The CLI’s allocator/RSS flags do not reproduce the campaign’s external whole-device/process-tree sampler by themselves.

## Exact builds and checkpoints

| Measured build | Source commit | Immutable image |
| --- | --- | --- |
| Completed pilot | [`2b37bb6888179a19fd8c02071b7cd4bc7bdfcf5f`](https://github.com/weertman/deepstarnoodles-inference/tree/2b37bb6888179a19fd8c02071b7cd4bc7bdfcf5f) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:48840df16d7b3e638eca357e34c045e4c4e84ae1951388974502ca65acd5e12b` |
| Completed Round 2 | [`e5dcb7c1040c8ba8be9c1d01a96698074981b36f`](https://github.com/weertman/deepstarnoodles-inference/tree/e5dcb7c1040c8ba8be9c1d01a96698074981b36f) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:fb2364ba364a0577a33607930caa0152c530fd43b0f311a91be290a9eb70bfec` |
| Completed Round 3 | [`fe68dfe3732b878a3520f1bc36e5058dae913612`](https://github.com/weertman/deepstarnoodles-inference/tree/fe68dfe3732b878a3520f1bc36e5058dae913612) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:931c8f145c2553d167bdc627ef6dba054adf9b5e500c537c19565762bd787e64` |
| Completed Round 4 | [`fe68dfe3732b878a3520f1bc36e5058dae913612`](https://github.com/weertman/deepstarnoodles-inference/tree/fe68dfe3732b878a3520f1bc36e5058dae913612) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:931c8f145c2553d167bdc627ef6dba054adf9b5e500c537c19565762bd787e64` |

Runtime: PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 91002, Ultralytics 8.4.51, NumPy 2.2.6, OpenCV 4.11.0, eight PyTorch CPU threads. Separate hosts/builds are not timing controls for one another. Independent comparisons established identical pilot/Round 2 and Round 2/Round 3 reference predictions on the same ordered 256 inputs and checkpoint hashes; that validates baseline output continuity only. Round 3 used a separate host. Its faster initial reference is not itself an optimization gain. Round 4 uses the same source/image as Round 3 on a separate lease and reproduces its eager reference bytes; only within-round pairs support timing claims.

| Checkpoint | SHA-256 |
| --- | --- |
| StarSeg, YOLO26-L segmentation | `5929d8a9c1ae3cf9a8b5c3ab9b1372221fe9f8d6cadd5a94f020aac8c9c3fb53` |
| Hydra, DINOv3 ConvNeXt-Base, EMA export | `aa056f933c1a0994fb299203aa20423c5208637d2eb4ede7e942abcc3aa91d7c` |
| Arm tracer, GRU hidden size 128 | `e6c1506067fc5864ab1d0d648dec2f7932632017d952753cb68f80fed2c4812f` |

The GRU traces arms within an image; live temporal tracking is separate. The [model release](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-seed0-20261028) supplies the portable manifest and checkpoints.

The completed measurements are bounded to the configurations and workload above. The numeric evidence preserves exact source/image pins, raw repeats, signed effects, resource coverage and verified final-manifest hashes.
