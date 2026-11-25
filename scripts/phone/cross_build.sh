#!/usr/bin/env bash
# Runs inside ps1-cross. /out receives llama_main + its libraries and the Arm64 Python package folder (site).
set -euo pipefail
JOBS="${JOBS:-12}"
export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
cd /opt/executorch
BUCK2="$(ls buck2-bin/buck2-*)"; export BUCK2
SRCS=/opt/executorch/cmake-out/executorch_srcs.cmake
CROSS=(-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
       -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
       -DFLATC_EXECUTABLE=/opt/executorch/cmake-out/third-party/flatbuffers/flatc -DBUCK2="$BUCK2"
       -DPYTHON_EXECUTABLE="$(command -v python)" -DCMAKE_BUILD_TYPE=Release)
KERNELS=(-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON)

echo "== Arm64 Python packages"
SITE=/out/site
PLATFORM=(--platform manylinux_2_28_aarch64 --platform manylinux_2_27_aarch64 --platform manylinux2014_aarch64
          --platform manylinux_2_17_aarch64 --platform any --python-version 3.12 --only-binary=:all: --no-deps)
pip install -q --target "$SITE" "${PLATFORM[@]}" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q --target "$SITE" "${PLATFORM[@]}" -r /kit/requirements-phone.txt

echo "== core runtime and llama_main"
L=/tmp/llama; mkdir -p "$L"; cp "$SRCS" "$L/"
cmake "${CROSS[@]}" "${KERNELS[@]}" -DEXECUTORCH_SRCS_FILE="$L/executorch_srcs.cmake" -DCMAKE_INSTALL_PREFIX="$L" \
  -DEXECUTORCH_ENABLE_LOGGING=1 -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON -B"$L" . > /out/llama_core.log 2>&1
cmake --build "$L" -j"$JOBS" --target install >> /out/llama_core.log 2>&1
cmake "${CROSS[@]}" "${KERNELS[@]}" -DEXECUTORCH_SRCS_FILE="$L/executorch_srcs.cmake" -DCMAKE_INSTALL_PREFIX="$L" \
  -DCMAKE_PREFIX_PATH="$L" -Dexecutorch_DIR="$L/lib/cmake/ExecuTorch" -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
  -B"$L/examples/models/llama" examples/models/llama > /out/llama_main.log 2>&1
cmake --build "$L/examples/models/llama" -j"$JOBS" >> /out/llama_main.log 2>&1
mkdir -p /out/pi-bin/lib
cp "$L/examples/models/llama/llama_main" /out/pi-bin/llama_main.bin
for lib in libllama_runner.so libextension_module.so; do cp "$(find "$L" -name "$lib" | head -1)" /out/pi-bin/lib/; done
aarch64-linux-gnu-strip /out/pi-bin/llama_main.bin /out/pi-bin/lib/*.so
cat > /out/pi-bin/llama_main <<'WRAPPER'
#!/bin/sh
HERE="$(cd "$(dirname "$0")" && pwd)"
LD_LIBRARY_PATH="$HERE/lib" exec "$HERE/llama_main.bin" "$@"
WRAPPER
chmod +x /out/pi-bin/llama_main

echo "== Python bindings"
P=/tmp/pyb; mkdir -p "$P"; cp "$SRCS" "$P/"
cmake "${CROSS[@]}" "${KERNELS[@]}" -DEXECUTORCH_SRCS_FILE="$P/executorch_srcs.cmake" -DEXECUTORCH_BUILD_PYBIND=ON \
  -DCMAKE_PREFIX_PATH="$SITE/torch" -DTorch_DIR="$SITE/torch/share/cmake/Torch" \
  -DTORCH_PYTHON_LIBRARY="$SITE/torch/lib/libtorch_python.so" -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_ENABLE_LOGGING=ON -B"$P" . > /out/pybind.log 2>&1
cmake --build "$P" -j"$JOBS" --target portable_lib >> /out/pybind.log 2>&1

# The wheel's Python files are the same on every CPU; only the compiled module is replaced by the Arm64 one.
cd "$SITE"
python - <<'PY'
import glob, zipfile
wheel = zipfile.ZipFile(glob.glob("/opt/wheel/*.whl")[0])
for name in wheel.namelist():
    if not name.endswith((".so", ".a")) and name != "executorch/data/bin/flatc":
        wheel.extract(name, ".")
PY
cp "$P"/_portable_lib.cpython-312-*.so executorch/extension/pybindings/_portable_lib.cpython-312-aarch64-linux-gnu.so
aarch64-linux-gnu-strip --strip-unneeded executorch/extension/pybindings/_portable_lib.cpython-312-aarch64-linux-gnu.so
for dir in torch/include torch/share; do find "$SITE/$dir" -delete; done                # build-time only
find "$SITE/torch/bin" -type f ! -name torch_shm_manager -delete                       # torch refuses to import without it
find "$SITE" -name '__pycache__' -prune -exec find {} -delete \;
aarch64-linux-gnu-readelf -h /out/pi-bin/llama_main.bin executorch/extension/pybindings/_portable_lib*.so | grep Machine
