# DeepStarNoodles inference

Portable StarSeg → Hydra → GRU128 inference for live replay and image archives. The wheel contains the runtime, crop/tracking configuration, and DINOv3 ConvNeXt source. It needs no research checkout. Checkpoints and images remain external.

The initial pinned generation uses YOLO26-L segmentation, **DINOv3 ConvNeXt-Base**, and a GRU with 128 hidden units. “128” is the arm-walker state width; the recurrence follows each arm within an image. Live replay adds BoT-SORT animal IDs and per-animal arm association. Archive images are independent. This package does not substitute a Large backbone for the trained Base checkpoint.

## Docker

Build from this directory. The version-pinned runtime is PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.8, and Ultralytics 8.4.51. CUDA 12.8 is selected for the RTX 5060 target. An allocator cap on a larger GPU does not reproduce that GPU's performance or guarantee total VRAM fit.

```bash
docker build -t deepstarnoodles-inference:0.1.0 .
docker run --rm --gpus all --memory=32g --cpus=8 --shm-size=1g \
  -v "$PWD/models:/models:ro" -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/outputs:/outputs" deepstarnoodles-inference:0.1.0 \
  archive --models /models --index /inputs/archive.csv \
  --output /outputs/archive-b4 --images 128 --warmup 8 \
  --detector-batch-size 4 --crop-batch-size 4
```

For one ordered stream, replace `archive` with `live-replay`, select its chronological CSV, and use `--detector-batch-size 1`. Crop micro-batches may combine animals in the same frame; frames stay causal. Start a separate command for each clip so tracker state resets. Output directories must be new.

`archive` and `live-replay` process every CSV row by default, with no separate warmup. An explicit `--images` limits the run. The `benchmark` command instead defaults to 128 measured images and eight warmup images; both forms share the same inference/output path.

`staged-archive` is the disk-spooled archive candidate: it runs the detector and dense/tracer stages separately to reduce simultaneous model residency. See `deepstarnoodles-inference staged-archive --help` for its arguments. Compare its whole-job time and spool storage with the resident archive path.

```bash
deepstarnoodles-inference staged-archive --models models/ --index inputs/archive.csv \
  --output outputs/staged --images 128 --detector-batch-size 4 --crop-batch-size 4 \
  --shard-size 32 --max-spool-gib 8
```

`--shard-size` bounds the number of source images processed before reloading stages; `--max-spool-gib` bounds live intermediate files and stops the run if exceeded. Stage initialization and spool I/O are included in job time. Staged archive defaults to archive mode with zero warmup; its default image limit is 128, so set `--images` explicitly for a full workload.

For RunPod's SSH worker mode, set the container argument to `runpod-worker`, expose TCP port 22, and supply your SSH **public** key in `PUBLIC_KEY`. The same image starts OpenSSH with public-key authentication and password login disabled; `rsync` is included for workload transfer. Host keys are created at container startup. No cloud tokens or private SSH keys belong in the image. Ordinary Docker invocations continue to run the CLI.

Use `--profile-stages` in a separate run for synchronized per-stage timings. Throughput runs omit it. Live replay measures processing of ordered files, including decoding and JSONL output; it does not simulate camera arrival, dropped-frame handling, or queue latency.

## Models and inputs

Place three trusted checkpoints and `manifest.json` in `models/`. `config/manifest.seed0.example.json` pins the initial model hashes; rename/copy your corresponding files to `starseg.pt`, `hydra.pt`, and `tracer.pt`. Relative model paths resolve against the manifest directory. `--models` accepts either this directory or a manifest file. The initial model selector uses the checkpoint's EMA export, and the tracer keeps its own normalization buffers and trace parameters. No older calibration is substituted.

```bash
deepstarnoodles-inference verify-models --models models/
```

Model weights are not committed or baked into the image. If you have authorized HTTPS download URLs, add a `url` to each model record and explicitly run `fetch-models --models manifest.json --destination models`. It verifies SHA-256 and removes download URLs from the resulting manifest. There are no implicit downloads during inference.

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
