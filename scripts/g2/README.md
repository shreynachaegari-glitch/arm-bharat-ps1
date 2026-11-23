# G2 bring-up: Raspberry Pi 5 (8 GB, CPU only)

Goal: the deploy candidate and the Llama model run offline on the Pi, with honest latency, memory and thermal numbers.
You need: Pi 5 8 GB, official 27 W PSU, active cooler, 64-bit Raspberry Pi OS Bookworm, internet for steps 1-3 only.
All commands run from the repo root on the Pi unless stated. Save every log; paste the results into the project notes.

## 0. Copy from the dev box
`-L` matters: `runs/bench/images` holds symlinks, and a plain copy brings dangling links. No sealed data goes across.
The leading `/` matters too: a bare `--exclude data` would also drop the code in `ps1/data/`.

    rsync -avL --exclude /data --exclude /logs --exclude '/runs/v[0-5]*' --exclude '/.*' \
      ~/hello-world/arm-bharat-ps1/ pi@<pi-address>:~/arm-bharat-ps1/

## 1. Tools
    sudo apt update && sudo apt install -y podman rsync
    curl -LsSf https://astral.sh/uv/install.sh | sh

## 2. Build the runtime (slow: allow 1-2 hours in total; keep the Pi cooled)
PyPI has no aarch64 wheel for executorch 0.6.0, so both the Python bindings and `llama_main` are built from the pinned commit.

    podman build -t ps1-pi-runtime:0.6.0 --build-arg JOBS=3 -f scripts/g2/Containerfile.pybind scripts/g2
    podman build -t ps1-et:0.6.0 --build-arg JOBS=3 -f scripts/g1/Containerfile scripts/g1

Take the two products out of the images. `llama_main` needs its libraries next to it, so use the script, not a plain copy:

    mkdir -p pi-bin
    podman run --rm -v "$PWD/pi-bin:/out:Z" ps1-pi-runtime:0.6.0 sh -c 'cp /opt/wheel/*.whl /out/'
    scripts/g2/extract_runner.sh pi-bin

## 3. Python 3.12 venv (Python 3.12 is required: locked numpy has no 3.11 wheel)
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r scripts/g2/requirements-pi.txt
    uv pip install --python .venv/bin/python --no-deps pi-bin/executorch-*.whl

## 4. Checks, in this order. Stop at the first failure.
Disconnect the network from here on: the app must work with none.

    scripts/g2/preflight.sh | tee logs/g2_preflight.log
    .venv/bin/python -m pytest -q | tee logs/g2_pytest.log
    .venv/bin/python -m ps1.vision.golden --check --pte runs/v6_final/export/tomato_effnet_b0_t2_fp32_xnnpack.pte \
      --golden runs/v6_final/golden.json --report logs/g2_golden.json

Golden check: same `.pte` and images as x86. `top1_agreement` must be 1.0. If only `max_abs_diff` is above the tolerance, or
`tensor_mismatches` is above 0, do not change anything: send the JSON to Sreyesh. Tensors differing means the image decode
differs on this board, which is a different finding from the runtime differing.

## 5. Benchmarks (idle Pi: nothing else running, no desktop if possible)
    export PS1_LLAMA_MAIN="$PWD/pi-bin/llama_main"
    .venv/bin/python -m ps1.bench --matrix scripts/g2/bench_matrix.json --images runs/bench/images --out runs/bench/pi5

Any line marked CONTAMINATED or LLAMA INVALID is not a result: fix the cause and rerun it. A Llama time of a few milliseconds
means the runner failed and the template answered. `runs/bench/pi5/*.json` also holds temperature and
throttling before and after. Note by hand: PSU, cooler, room temperature. Target: Screen-1 p95 at most 1.5 s.

## 6. Llama through the G1 runner (10 runs: load, first token, tok/s, RSS)
    scripts/g1/run_g1.sh 10 4
    scripts/g1/run_g1.sh 10 2

## 7. Optional cross-check against llama.cpp (speed and memory only; Sreyesh decides whether to download the GGUF)
Pinned file: `bartowski/Llama-3.2-1B-Instruct-GGUF` rev `067b946cf014b7c697f3654f621d577a3e3afd1c`,
`Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807,694,464 B, sha256 `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`
(Llama 3.2 licence). The weights differ from our SpinQuant `.pte`, so this never says which model answers better.

    podman build -t ps1-llamacpp:cd959ba --build-arg JOBS=3 -f scripts/g2/Containerfile.llamacpp scripts/g2
    scripts/g2/run_llamacpp_baseline.sh <path>/Llama-3.2-1B-Instruct-Q4_K_M.gguf 10 4 logs/<g1_run_t4>/g1_runs.csv

## If something goes wrong
- Compiler killed during a build: out of memory. Rebuild with `--build-arg JOBS=2`.
- `get_throttled` not `0x0`: PSU or cooling problem. Fix it and reboot; earlier numbers do not count.
- `libllama_runner.so: cannot open shared object file`: `llama_main` was copied by hand. Use `scripts/g2/extract_runner.sh`.
- `llama_main` fails with a GLIBC error: the OS is older than the build image (Debian Bookworm). Use Pi OS Bookworm or newer.
- Slow vision, about 1 s per image: another process is using the cores (XNNPACK uses every core). Check `top`.
