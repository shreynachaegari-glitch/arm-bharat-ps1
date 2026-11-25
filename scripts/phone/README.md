# Running on an Arm Android phone (no root, no internet at run time)

The bundle holds an Arm64 Debian with Python 3.12, ExecuTorch 0.6.0 cross-compiled from the pinned commit, both models
and the app. `proot` runs it inside Termux, so nothing is compiled or installed system-wide on the phone.
Needs Android 12 or newer on arm64, 6 GB RAM or more and about 4 GB free storage.

## 1. Build the bundle (x86 dev box)
Needs the two images from `scripts/g1/Containerfile` (`ps1-et:0.6.0`) and `scripts/g2/Containerfile.pybind`
(`ps1-pi-runtime:0.6.0`).

    scripts/phone/build_bundle.sh          # writes build/phone/ps1-phone.tar.gz and its sha256

## 2. Prepare the phone (internet needed for this step only)
1. Install **Termux** from F-Droid or its GitHub releases page (not the Play Store build).
2. Settings > About phone > tap the build number seven times, then in Developer options turn on
   **Disable child process restrictions** (Android 14+) and restart. Without it Android may kill the app (signal 9).
3. Settings > Apps > Termux > Battery > Unrestricted.
4. In Termux: `pkg update -y && pkg install -y proot tar` and then `termux-setup-storage` (allow).

## 3. Install
Copy `ps1-phone.tar.gz` to the phone's **Download** folder over USB, then in Termux:

    sha256sum ~/storage/downloads/ps1-phone.tar.gz      # compare with ps1-phone.tar.gz.sha256
    tar -xzf ~/storage/downloads/ps1-phone.tar.gz -C ~
    cd ~/ps1-phone

## 4. Check, then measure
Plug in, close other apps, keep the screen on.

    ./bench.sh

The golden check must say `"passed": true`: same model, same images, same answers as the x86 reference. Latency, memory
and the 10 Llama runs land in `app/results/`. Android hides the CPU load from apps, so the benchmark says it cannot tell
whether the phone was idle; that is why other apps must be closed.

## 5. Use
    ./start.sh

Turn on aeroplane mode and open `http://127.0.0.1:5000` in the phone's browser. Stop with Ctrl+C in Termux.

## If something goes wrong
- `Process completed (signal 9)`: step 2.2 was skipped, or memory is short: close other apps.
- `proot: command not found`: step 2.4.
- The explanation never appears: Llama took more than 60 s; the reviewed text stays on screen. See `results/llama_t4`.
