#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: bash %s [--check-only] INFINILM_REPO MODEL_PATH OUTPUT_DIR\n' "$0" >&2
  exit 2
}

check_only=false
if [[ ${1:-} == --check-only ]]; then
  check_only=true
  shift
fi
[[ $# == 3 ]] || usage

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$1" && pwd -P)
model_dir=$(cd -- "$2" && pwd -P)
mkdir -- "$3" || die 'Output directory must be new and its parent must exist.'
output_dir=$(cd -- "$3" && pwd -P)
python_bin=${PYTHON_BIN:-python}
first_frame="$bundle_dir/first_frame.png"
generator="$repo_dir/examples/minimax_h3.py"
comparator="$repo_dir/examples/minimax_h3_compare.py"
expected_gpus=0,1,2,3,4,5,6,7

[[ -s "$first_frame" ]] || die 'Missing bundled first_frame.png.'
[[ -s "$generator" && -s "$comparator" ]] || die 'MiniMax-H3 scripts are missing from InfiniLM.'
[[ -s "$model_dir/model_index.json" || -s "$model_dir/diffusers/model_index.json" ]] || \
  die 'Model path must point to a local MiniMax-H3 diffusers directory or its parent.'
if [[ -n ${REFERENCE_FL2VA_VIDEO:-} ]]; then
  [[ $REFERENCE_FL2VA_VIDEO = /* && -s $REFERENCE_FL2VA_VIDEO ]] || \
    die 'REFERENCE_FL2VA_VIDEO must be an absolute path to a nonempty official reference video.'
fi

cd -- "$repo_dir"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
{
  printf 'run_started_utc=%s\nrepo=%s\nmodel=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$repo_dir" "$model_dir"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git log -1 --format='infinilm_commit=%H'
    git status --short
  fi
  "$python_bin" - "$first_frame" "$generator" "$comparator" <<'PY'
import hashlib
import sys
from pathlib import Path

import av
import infinicore
import infinilm
import torch
from PIL import Image
from skimage.metrics import structural_similarity

print(f"python={sys.executable} version={sys.version.split()[0]}")
print(f"torch={torch.__version__} hip={torch.version.hip} av={av.__version__}")
print(f"infinicore={infinicore.__file__}\ninfinilm={infinilm.__file__}")
print(f"infinicore.use_ntops={getattr(infinicore, 'use_ntops', 'not exposed')}")
for name in sys.argv[1:]:
    path = Path(name)
    print(f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()} {path.name}")
with Image.open(sys.argv[1]) as frame:
    print(f"first_frame={frame.size} format={frame.format}")
    frame.verify()
if not torch.version.hip or not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot use the required HIP devices.")
count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"Expected 8 visible DCUs, found {count}.")
for index in range(count):
    device = torch.cuda.get_device_properties(index)
    arch = getattr(device, "gcnArchName", "unknown")
    print(f"device={index} name={device.name} arch={arch} vram_GiB={device.total_memory / 2**30:.2f}")
    value = torch.ones(1, device=f"cuda:{index}").sum().item()
    if value != 1:
        raise SystemExit(f"Device {index} failed the basic PyTorch operation.")
PY
  "$python_bin" "$generator" fl2va --help >/dev/null
  "$python_bin" "$comparator" --help >/dev/null
  printf 'FL2VA environment checks completed.\n'
} 2>&1 | tee "$output_dir/environment.log"

if "$check_only"; then
  exit 0
fi

video="$output_dir/infinilm_fl2va.mp4"
"$python_bin" "$generator" fl2va \
  --model-path="$model_dir" --first-frame="$first_frame" \
  --text='A dog falls asleep' --gpus="$expected_gpus" \
  --seed=42 --num-frames=124 --steps=50 --output="$video" \
  2>&1 | tee "$output_dir/infinilm_fl2va.log"
[[ -s "$video" ]] || die 'InfiniLM did not produce a nonempty video.'

compare_args=(--tasks=fl2va --model-path="$model_dir" --first-frame="$first_frame"
  --output-dir="$output_dir" --infinilm-fl2va-video="$video"
  --legacy-gpus="$expected_gpus" --legacy-memory-fraction="${LEGACY_MEMORY_FRACTION:-0.4}")
if [[ -n ${REFERENCE_FL2VA_VIDEO:-} ]]; then
  compare_args+=(--reference-fl2va-video="$REFERENCE_FL2VA_VIDEO")
fi
comparison_log="$output_dir/compare_fl2va.log"
"$python_bin" "$comparator" "${compare_args[@]}" 2>&1 | tee "$comparison_log"
grep -Eq '^[[:space:]]+Size:[[:space:]]+PASS' "$comparison_log" || die 'Comparison did not pass the video size check.'
grep -Eq '^[[:space:]]+Length:[[:space:]]+PASS \(PyTorch 124 frames, InfiniLM 124 frames\)' "$comparison_log" || \
  die 'Comparison did not confirm that both videos contain 124 frames.'
grep -Eq '^[[:space:]]+Average SSIM:[[:space:]]+-?[0-9]+(\.[0-9]+)?%' "$comparison_log" || \
  die 'Comparison did not report a numeric average SSIM.'
