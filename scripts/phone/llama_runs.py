"""Run llama_main N times on the G1 prompt and summarise it like scripts/g1 (no container, no GNU time needed).

Usage: python scripts/phone/llama_runs.py <runs> <threads> <out_dir>
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 300


def run_once(command: list[str], log: Path) -> None:
    """Write the runner's output plus the two lines parse_g1.py reads from GNU time (peak RSS, exit status)."""
    with log.open("wb") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        # wait4 gives this child's own peak memory; getrusage(RUSAGE_CHILDREN) would give the largest child so far.
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
        out.write(f"\nMaximum resident set size (kbytes): {usage.ru_maxrss}\nExit status: {process.returncode}\n".encode())


def main() -> None:
    runs, threads, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    runner = os.environ["PS1_LLAMA_MAIN"]
    models = ROOT / "artifacts/llama"
    prompt = (ROOT / "scripts/g1/prompt.txt").read_text()
    (out / "prompt.txt").write_text(prompt)
    command = ["timeout", str(TIMEOUT_S), runner, f"--model_path={models}/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte",
               f"--tokenizer_path={models}/tokenizer.model", f"--prompt={prompt}", "--seq_len=512", "--temperature=0",
               f"--cpu_threads={threads}"]
    for i in range(1, runs + 1):
        print(f"run {i}/{runs}", flush=True)
        run_once(command, out / f"run_{i}.log")
    subprocess.run([sys.executable, str(ROOT / "scripts/g1/parse_g1.py"), str(out)], check=True)


if __name__ == "__main__":
    main()
