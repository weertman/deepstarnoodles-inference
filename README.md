# DeepStarNoodles inference

Portable StarSeg → Hydra → GRU128 inference for video files, image archives, cameras and live replay. The wheel contains the runtime, crop and tracking configuration, and DINOv3 ConvNeXt source, so it needs no research checkout. Checkpoints and images stay external.

The trained models are YOLO26-L segmentation, **DINOv3 ConvNeXt-Base** and a GRU with 128 hidden units, which follows each arm within an image. Live modes add BoT-SORT animal IDs and per-animal arm association. Archive images are independent.

| Document | What it covers |
|---|---|
| This README | Installing, getting the models and the commands to run |
| [docs/options.md](docs/options.md) | Every speed, scheduling and live option, including the ones measurement rejected |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | Every measurement, with rates per GPU type, output agreement and memory |

**Version note.** This branch is 0.5.0.dev10. The published container image is still 0.4.0, so it lacks `serve-videos` and most options in the commands below. Install from source or build the image from this repository, as shown next.

## Install

Use Python 3.11 or newer. Install the target PyTorch wheels first, then the package:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-cuda128.txt -r requirements-runtime.txt
python -m pip install .
deepstarnoodles-inference smoke --device cuda:0
```

The smoke command builds the full architecture with random weights and runs a small synthetic forward pass. It checks the installation, not accuracy or speed. `--device cpu` works for development.

Or build the container, which pins PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.8 and Ultralytics 8.4.51:

```bash
docker build -t deepstarnoodles-inference .
```

Inside the container the same commands run as `docker run --rm --gpus all --shm-size=1g -v "$PWD:/work" deepstarnoodles-inference <command>`, without the `deepstarnoodles-inference` prefix. For RunPod's SSH worker mode, set the container argument to `runpod-worker`, expose TCP port 22, and supply your SSH **public** key in `PUBLIC_KEY`. Host keys are created at startup, and password login is disabled.

## Get the models

The checkpoints are published in [models-seed0-20261028](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-seed0-20261028), about 793 MB in total. This downloads them and verifies their SHA-256 hashes:

```bash
curl -fL -o manifest.json \
  https://github.com/weertman/deepstarnoodles-inference/releases/download/models-seed0-20261028/manifest.json
deepstarnoodles-inference fetch-models --models manifest.json --destination models
deepstarnoodles-inference verify-models --models models/
```

`models/` must not already exist. Nothing is downloaded implicitly during inference. Checkpoint loading uses PyTorch serialization, so only load trusted weights whose hashes match the manifest.

## Annotate videos

`serve-videos` is a resident service for annotating many videos. This is the configuration measured across GPU types:

```bash
nvidia-cuda-mps-control -d
deepstarnoodles-inference serve-videos --queue queue --output out --models models/manifest.json --processes 8 \
  --device cuda:0 --detector-batch-size 1 --crop-batch-size 1 --hydra-output-mode minimal \
  --gru-execution graph --gru-check-interval 1 --hydra-execution graph \
  --detect-ahead 2 --decode-prefetch-frames 4 --gpu-crops --cuda-blocking-sync \
  --hydra-fp16-weights --hydra-compile --gru-compile --gru-graph-first-step
```

Move each finished video into `queue/ready/` with a rename. Each worker claims whole videos, decodes every frame and annotates each frame independently. Records are appended to `out/<video>.jsonl.part`, which becomes `out/<video>.jsonl` when the video is done, and the video is then deleted, so the queue works as a rolling cache. A `STOP` file in the queue directory ends every worker after its current frame, and undecodable videos move to `queue/failed/`. `--claim-prefix P` limits workers to names starting with `P`.

On eight GPU types this sustained 19–67 images/s per GPU on a mix of lab videos with about 2.8 stars per frame. The RTX 4090 cost the least per frame, at 41 images/s. [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md#round-16-speed-test-across-gpu-types) has the table.

- **Always run several processes under CUDA MPS** (see [Known issues](#known-issues)). Keep `CUDA_MPS_PIPE_DIRECTORY` short, because the control socket path must fit the 108-byte Unix socket limit.
- **Six to ten processes** saturated every GPU tested. Each holds about 2 GiB of GPU memory, flat from video to video, at every resolution tested up to 5496×3672.
- **`--allocator-limit-gib`** defaults to 7 per process, so lower it if the processes together could fill the card. Setting it to `(0.9 × GPU MiB ÷ processes − 400) ÷ 1024` leaves room for each process's CUDA context, and 1.6 was enough in testing.

Compiled outputs are not byte-identical to the uncompiled pipeline, but the difference is very small. On the 478-image validation split F1 is 0.8033 against 0.8034, with one extra false positive in 4,801 arms. On the lab mix 99.6% of arms stay within 2 px, and star counts match in all 371 frames. For byte-identical outputs, leave out `--hydra-compile --gru-compile`, which cost about a quarter of the speed in a local test.

## Annotate images

`archive` annotates a CSV of images with an `image_path` column; relative paths resolve against the CSV. Under MPS, this configuration reached about 52 images/s on an RTX 5090, and its predictions are byte-identical to the earlier Round 4 selection, which passed the strict parity gates against the reference:

```bash
nvidia-cuda-mps-control -d
deepstarnoodles-inference archive --models models/ --index inputs/archive.csv --output outputs/archive \
  --detector-batch-size 1 --crop-batch-size 1 --hydra-output-mode minimal \
  --gru-execution graph --gru-check-interval 1 --hydra-execution graph \
  --detect-ahead 2 --archive-processes 8 --gpu-crops --cuda-blocking-sync \
  --decode-workers 2 --decode-prefetch-frames 4 --decode-prefetch-mib 256 --decode-max-frame-mib 64
```

Output directories must be new. Results are `predictions.jsonl` and a `report.json` with checkpoint hashes, hardware, runtime details, crop counts, timing and memory peaks. Images with no detections still get a record.

## Annotate a camera or a live stream

```bash
deepstarnoodles-inference live --models models/ --camera 0 --output outputs/camera \
  --max-seconds 60 --crop-batch-size 1 --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1
deepstarnoodles-inference live --models models/ --video inputs/clip.mp4 --output outputs/clip \
  --crop-batch-size 1 --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1
```

Camera mode keeps one pending frame and reports dropped frames. A video is processed frame by frame in order, and it doesn't simulate camera timing. `live-replay` does the same for a chronological CSV of images, one stream per command. In Docker, add `--device=/dev/video0:/dev/video0` to use a camera.

The same resident pipeline accepts frames from your own code:

```python
from deepstarnoodles_inference.live import LiveConfig, LivePipeline

pipeline = LivePipeline.from_manifest("models/", LiveConfig(device="cuda:0", crop_batch_size=1))
result = pipeline.predict(frame_bgr)  # H x W x 3 numpy.uint8, BGR order
pipeline.reset()  # New independent stream; models stay loaded.
```

Call `predict()` synchronously and don't change the frame until it returns. After a failed inference, call `reset()` before continuing.

Latency on short warmed replays improved with this profile, but those replays don't establish sustained camera frame rates. Test it on your own stream. [docs/options.md](docs/options.md#camera-and-live-details) has more on timing and shutdown.

## Output format

Each line of a `.jsonl` file is one frame:

- A frame has `image_path`, `source_shape` and `stars`, plus `video` and `frame` from `serve-videos`.
- A star has `detection_index`, `star_track_id`, `crop` (`left`, `top`, `side`, `confidence`, `bbox_xyxy`), `mask_polygon_source` and `arms`.
- An arm has `polyline_source`, `polyline_crop`, `length_px`, `steps`, `stop_reason`, `state_confidence`, `tip_score`, `base_score` and an `identity` vector. Archive arms use `arm_index`, and live arms use `track_id` scoped to their `star_track_id`.

Coordinates are in source pixels unless the name says crop.

## Known issues

- **Worker faults without MPS.** When several processes time-slice one GPU, graph execution with `--detect-ahead` occasionally aborts a process with a CUDA illegal memory access (Xid 31, a read fault through the constant cache). `serve-videos` without MPS faulted 3 times in about 1.1 process-hours. Under CUDA MPS none has occurred in about 45 streamed process-hours on rented pods, or in the archive rounds. The root cause has not been isolated. A fault ends the process, and no run has produced different predictions. Under MPS a fatal fault in one client ends them all, so wrap long jobs in a retry.
- **Outputs vary slightly by GPU chip family.** Runs on one GPU type repeat byte for byte. Between Ampere, Ada and Blackwell, about 1.5% of arms lack a match within 2 px and 0.2–0.3% of frames change star count. Star counts, arm counts and arm lengths agree within 0.1%, so it looks like noise rather than bias.
- **Larger batches fail output parity.** Keep detector and crop batches at one.
- **No RTX 5060 or 8 GB measurement exists.** An allocator limit on a larger GPU doesn't reproduce a smaller GPU's speed or guarantee that it fits.

## Compare two runs

`compare` streams two prediction files through fixed parity gates. It needs no GPU or model weights:

```bash
deepstarnoodles-inference compare --reference outputs/a/predictions.jsonl \
  --candidate outputs/b/predictions.jsonl --output outputs/comparison.json
```

It exits 0 only for a pass, 1 for a failed comparison or a file error, and 2 for invalid arguments. `--output` must name a new file. Source paths must match, so mount inputs at the same path in both runs. A pass doesn't establish accuracy, memory fit or a speedup.

## Development

```bash
python -m pip install ".[test]"
python -m pytest
```

Tests that need CUDA skip themselves on CPU. The GitHub Actions workflow tests the installed wheel on CPU and builds the CUDA image on every pull request. It publishes `ghcr.io/weertman/deepstarnoodles-inference` on pushes to `main`, version tags and manual dispatch, and it records the immutable image digest. Use that digest for experiments. A commit with `[skip ci]` in its message updates `main` without publishing an image.

Licensing: project runtime code is MIT; bundled DINOv3 and BoT-SORT material retains its own terms. See `THIRD_PARTY_NOTICES.md` and `licenses/`.
