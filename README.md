# DeepStarNoodles inference

Portable StarSeg → Hydra → GRU128 inference for cameras, video files, live replay, and image archives. The wheel contains the runtime, crop/tracking configuration, and DINOv3 ConvNeXt source. It needs no research checkout. Checkpoints and images remain external.

The initial pinned generation uses YOLO26-L segmentation, **DINOv3 ConvNeXt-Base**, and a GRU with 128 hidden units. “128” is the arm-walker state width; the recurrence follows each arm within an image. Live replay adds BoT-SORT animal IDs and per-animal arm association. Archive images are independent. This package does not substitute a Large backbone for the trained Base checkpoint.

## Download and run

The trained checkpoints are published in [models-seed0-20261028](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-seed0-20261028). Download the manifest, then let the container fetch and SHA-256-verify all three files. This needs about 793 MB for the checkpoint files, plus the CUDA image.

```bash
mkdir -p deepstarnoodles-run
cd deepstarnoodles-run
curl -fL -o manifest.json \
  https://github.com/weertman/deepstarnoodles-inference/releases/download/models-seed0-20261028/manifest.json

# Pull the public image, then preserve its immutable digest.
IMAGE=ghcr.io/weertman/deepstarnoodles-inference:latest
docker pull "$IMAGE"
# Save the immutable registry digest for subsequent runs.
IMAGE=$(docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}')
printf '%s\n' "$IMAGE" > image.txt

docker run --rm -v "$PWD:/work" "$IMAGE" \
  fetch-models --models /work/manifest.json --destination /work/models
```

`models/` must not already exist. The download command verifies every file and writes a portable local manifest. Fetching weights needs no GPU; inference requires an NVIDIA driver and Docker GPU support.

For an archive, put an `archive.csv` containing an `image_path` column in `inputs/`; paths resolve relative to that CSV. Start conservatively with one detector image and one crop per batch, then benchmark larger batches:

```bash
mkdir -p inputs outputs
docker run --rm --gpus all --memory=32g --cpus=8 --shm-size=1g \
  -v "$PWD/models:/models:ro" -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/outputs:/outputs" "$IMAGE" \
  archive --models /models --index /inputs/archive.csv \
  --output /outputs/archive --detector-batch-size 1 --crop-batch-size 1
```

For a local camera, add `--device=/dev/video0:/dev/video0` to the Docker options and replace the final command with `live --models /models --camera 0 --output /outputs/camera --max-seconds 60`. Camera mode keeps one pending frame and reports dropped frames. `live --video /inputs/clip.mp4` processes a file sequentially. Each output directory must be new. The shipped defaults are reference settings; publication alone does not establish 8 GB fit or a live frame-rate guarantee.

## Docker

Build from this directory. The version-pinned runtime is PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.8, and Ultralytics 8.4.51. CUDA 12.8 is selected for the RTX 5060 target. An allocator cap on a larger GPU does not reproduce that GPU's performance or guarantee total VRAM fit.

```bash
docker build -t deepstarnoodles-inference:0.2.1 .
docker run --rm --gpus all --memory=32g --cpus=8 --shm-size=1g \
  -v "$PWD/models:/models:ro" -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/outputs:/outputs" deepstarnoodles-inference:0.2.1 \
  archive --models /models --index /inputs/archive.csv \
  --output /outputs/archive-b4 --images 128 --warmup 8 \
  --detector-batch-size 4 --crop-batch-size 4
```

For one ordered stream, replace `archive` with `live-replay`, select its chronological CSV, and use `--detector-batch-size 1`. Crop micro-batches may combine animals in the same frame; frames stay causal. Start a separate command for each clip so tracker state resets. Output directories must be new.

`archive` and `live-replay` process every CSV row by default, with no separate warmup. An explicit `--images` limits the run. The `benchmark` command instead defaults to 128 measured images and eight warmup images; both forms share the same inference/output path.

`--hydra-output-mode minimal` is an opt-in memory candidate for all three commands and `staged-archive`. Archives skip the unused activated Hydra output; live replay activates and transfers only support and mouth maps for arm association. The default is `full`. Raw GRU inputs and tracking thresholds are unchanged. For 512-pixel float32 outputs, this removes an 18 MiB activation tensor per crop in archive mode and replaces it with 2 MiB per crop in live mode; actual peak VRAM and speed still require GPU measurement. The Python `predict_tensor_batch` API continues to return raw and full activated fields.

`staged-archive` is the disk-spooled archive candidate: it runs the detector and dense/tracer stages separately to reduce simultaneous model residency. See `deepstarnoodles-inference staged-archive --help` for its arguments. Compare its whole-job time and spool storage with the resident archive path.

```bash
deepstarnoodles-inference staged-archive --models models/ --index inputs/archive.csv \
  --output outputs/staged --images 128 --detector-batch-size 4 --crop-batch-size 4 \
  --shard-size 32 --max-spool-gib 8
```

`--shard-size` bounds the number of source images processed before reloading stages; `--max-spool-gib` bounds live intermediate files and stops the run if exceeded. Stage initialization and spool I/O are included in job time. Staged archive defaults to archive mode with zero warmup; its default image limit is 128, so set `--images` explicitly for a full workload.

For RunPod's SSH worker mode, set the container argument to `runpod-worker`, expose TCP port 22, and supply your SSH **public** key in `PUBLIC_KEY`. The same image starts OpenSSH with public-key authentication and password login disabled; `rsync` is included for workload transfer. Host keys are created at container startup. No cloud tokens or private SSH keys belong in the image. Ordinary Docker invocations continue to run the CLI.

Use `--profile-stages` in a separate run for synchronized per-stage timings. Throughput runs omit it. Live replay measures processing of ordered files, including decoding and JSONL output; it does not simulate camera arrival, dropped-frame handling, or queue latency.

## Camera, video, and ndarray input

The `live` command consumes BGR frames directly without intermediate image files or CSVs:

```bash
deepstarnoodles-inference live --models models/ --camera 0 \
  --output outputs/camera-session --crop-batch-size 2 --max-seconds 60
deepstarnoodles-inference live --models models/ --video inputs/clip.mp4 \
  --output outputs/video-session --crop-batch-size 2
```

Camera capture uses a background reader with **one pending latest frame**. If inference falls behind, a new frame replaces that pending frame; overwrite and shutdown discard counts appear in the report. `--queue-policy sequential` disables the background reader. Video files always use sequential capture so each decoded frame is processed once; unpaced video decoding cannot stand in for real camera arrival. The detector and Hydra/GRU stay resident, and crop batches combine animals within the current frame. `--max-frames` counts processed frames. Each new command starts fresh tracker state.

`--hydra-output-mode minimal` is an opt-in allocation/transfer candidate shared by `live`, replay, and archive commands. It retains the same raw Hydra fields for GRU inference, and live tracking receives only its support and mouth maps. The default `full` path remains available for comparison. The Python API selects this with `LiveConfig(hydra_output_mode="minimal")`; compare numerical output and timings before promoting it on a target GPU.

`--gru-execution fixed --gru-check-interval 8` enables an experimental GRU decoder in the same commands, including staged archives. It keeps one slot per seed and masks completed walkers; it checks whether all walkers have stopped every N steps (1–128). The reference decoder remains `--gru-execution dynamic`, and both settings appear in the report's configuration. The Python API uses `LiveConfig(gru_execution="fixed", gru_check_interval=8)`. Fixed slots reduce dynamic compaction and host synchronization, but completed slots still execute neural layers and can increase work. CPU numerical parity tests cover termination, filtering, batching, and real synthetic neural layers; GPU roundoff, performance, and memory fit must be measured before selecting this mode. No trained-image evidence is shipped in the package.

`source_index` counts successful capture reads; `processed_index` counts inference calls. Temporal trackers age by processed observations even when the source sequence skips. Reported acquisition time is the monotonic timestamp immediately after OpenCV `read()` returns, **not sensor exposure time**. Queue wait, inference completion, and total post-acquisition latency are measured separately; backend buffering and later JSONL write/flush are outside those latency values. Video timestamps are best-effort OpenCV `POS_MSEC`. Timing means cover the full session; quantiles explicitly cover the most recent 1,024 frames, keeping memory bounded. Cold first inference is included.

EOF, Ctrl-C, SIGTERM, and inference errors close the capture and write a final report. OpenCV cannot always interrupt a native blocking read: shutdown waits up to one second and reports `reader_stopped: false` and non-final capture statistics if it is still blocked. Deadlines are checked between operations. A video read failure may be either EOF or decoding failure; the output names that ambiguity. Camera access and target RTX 5060 throughput still require validation on the deployment machine.

For Docker camera access, add `--device=/dev/video0:/dev/video0` to the GPU Docker command, then use `live --camera 0`. For files, mount the input video read-only and use its container path.

The same resident API accepts application-owned frames:

```python
from deepstarnoodles_inference.live import LiveConfig, LivePipeline

pipeline = LivePipeline.from_manifest("models/", LiveConfig(device="cuda:0", crop_batch_size=2))
result = pipeline.predict(frame_bgr)  # H x W x 3 numpy.uint8, BGR order
# Call again for subsequent frames from this stream. No file writes are needed.
pipeline.reset()  # New independent stream; models remain loaded.
```

Call `predict()` synchronously and do not mutate its input until it returns. An external acquisition loop may pass strictly increasing `source_index`, `acquired_monotonic_ns=time.monotonic_ns()` recorded after acquisition, and an optional `source_timestamp_s`. Reset after a failed inference before continuing, because tracking may already have advanced. Concurrent calls on one pipeline are rejected.

## Models and inputs

Place three trusted checkpoints and `manifest.json` in `models/`. `config/manifest.seed0.example.json` pins the initial model hashes; rename/copy your corresponding files to `starseg.pt`, `hydra.pt`, and `tracer.pt`. Relative model paths resolve against the manifest directory. `--models` accepts either this directory or a manifest file. The initial model selector uses the checkpoint's EMA export, and the tracer keeps its own normalization buffers and trace parameters. No older calibration is substituted.

```bash
deepstarnoodles-inference verify-models --models models/
```

Model weights are public release assets and are not committed or baked into the image. The example manifest includes their pinned download URLs. Explicitly run `fetch-models --models config/manifest.seed0.example.json --destination models` to download and verify them. The destination must be new. The command removes download URLs from the resulting local manifest. There are no implicit downloads during inference.

Input CSVs require an `image_path` column. Relative image paths resolve against the CSV directory; absolute paths are accepted. A historical archive can be arbitrarily ordered. Each live CSV must contain one chronological stream. Outputs are streamed to `predictions.jsonl` plus a `report.json` containing checkpoint hashes, hardware/runtime details, crop counts, timing, and memory peaks. Empty detections still receive a source record. Archive arms use `arm_index`; live arms use `track_id` scoped to their `star_track_id`.

## Local installation and checks

Use Python 3.11 or newer. Install the target PyTorch wheels first, then the package:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-cuda128.txt -r requirements-runtime.txt
python -m pip install .
deepstarnoodles-inference smoke --device cpu
deepstarnoodles-inference smoke --device cuda:0
```

The smoke command constructs the full Hydra architecture and GRU128 with random weights, executes a small synthetic forward pass, and checks packaged resources. It establishes installation/runtime compatibility, not pretrained accuracy or target throughput. CPU wheels can be used for development. Install `.[test]` and run `pytest` for regression tests.

No RTX 5060 performance claim follows from shipping this package. Promote a configuration only after representative quality checks, sustained memory measurements, and measurements on the intended GPU. Checkpoint loading uses PyTorch serialization; only load trusted weights whose hashes match the manifest.

Licensing: project runtime code is MIT; bundled DINOv3 and BoT-SORT material retains its own terms. See `THIRD_PARTY_NOTICES.md` and `licenses/`.

The GitHub Actions workflow tests the installed wheel on CPU, builds the actual CUDA image, and publishes `ghcr.io/weertman/deepstarnoodles-inference` on main pushes, version tags, or manual dispatch. Pull requests build without publishing. The workflow records the immutable image digest; use that digest for experiments. Publication uses the repository's built-in `GITHUB_TOKEN`. The registry package must be public before an unauthenticated RunPod deployment can pull it.
