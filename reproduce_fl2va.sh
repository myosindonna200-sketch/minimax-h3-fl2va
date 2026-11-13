#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: bash reproduce_fl2va.sh RUN_ID prepare [ABSOLUTE_MODEL_DIR]' >&2
  echo '       bash reproduce_fl2va.sh RUN_ID core|lm|softmax|generate|decode|compare' >&2
  exit 2
}

[[ $# -ge 2 && $1 =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || usage
run_id=$1
stage=$2
case "$stage" in
  prepare) [[ $# == 2 || ( $# == 3 && $3 == /* ) ]] || usage ;;
  core|lm|softmax|generate|decode|compare) [[ $# == 2 ]] || usage ;;
  *) usage ;;
esac

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$project"
afs=${IFAB_AFS_HOME:?InfiniFabric personal storage is required}
work="$afs/minimax-h3-work"
run="$work/reproduce/$run_id"
mkdir -p "$run"

fail() { echo "ERROR: $*" >&2; exit 1; }
value() {
  [[ -s "$run/$1" ]] || fail "Run the prerequisite stage first: $1"
  cat "$run/$1"
}
save() {
  local pending
  pending=$(mktemp "$run/$1.XXXXXX")
  printf '%s\n' "$2" > "$pending"
  mv -f -- "$pending" "$run/$1"
}
last_value() {
  local output
  output=$(sed -n "s/^$1=//p" "$log" | tail -n 1)
  [[ -n $output ]] || fail "Missing $1 in $log"
  printf '%s\n' "$output"
}

if [[ -s "$run/$stage" ]]; then
  printf 'STAGE_COMPLETE=%s\n' "$stage"
  cat "$run/$stage"
  exit 0
fi

if [[ $stage == prepare ]]; then
  if [[ $# == 3 ]]; then
    model_dir=$3
    python - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / 'model_index.json').open() as index:
    if json.load(index).get('_class_name') != 'MiniMaxH3ModularPipeline':
        raise SystemExit(f'Expected MiniMax-H3 Diffusers weights under {root}')
for component in ('processor', 'tokenizer', 'text_encoder', 'transformer', 'vae',
                  'audio_vae', 'scheduler', 'audio_scheduler'):
    if not (root / component).is_dir():
        raise SystemExit(f'Missing MiniMax-H3 Diffusers component: {root / component}')
PY
  fi
  test -s first_frame.png
  (cd "$afs" && sha256sum -c "$project/offline/SHA256SUMS")
  bootstrap_archive="$afs/minimax-h3-offline-bootstrap.tar.gz"
  bootstrap_dir="$afs/h3-offline-bootstrap"
  if [[ ! -d "$bootstrap_dir" ]]; then
    (
      staging=$(mktemp -d "$afs/.h3-bootstrap.XXXXXX")
      trap 'rm -rf -- "$staging"' EXIT
      tar -xmzf "$bootstrap_archive" -C "$staging"
      (cd "$staging/h3-offline-bootstrap" && sha256sum -c SHA256SUMS)
      python -c 'import os, sys; os.rename(sys.argv[1], sys.argv[2])' \
        "$staging/h3-offline-bootstrap" "$bootstrap_dir"
    )
  else
    echo 'Checking existing h3-offline-bootstrap in personal AFS'
    if ! tar -xOzf "$bootstrap_archive" h3-offline-bootstrap/SHA256SUMS |
      (cd "$bootstrap_dir" && sha256sum -c -); then
      fail "Existing bootstrap failed verification: $bootstrap_dir"
    fi
  fi
  if [[ $# == 3 ]]; then save model-dir "$model_dir"; fi
  save prepare OK
  echo 'STAGE_COMPLETE=prepare'
  exit 0
fi

if [[ -s "$run/model-dir" ]]; then
  MINIMAX_H3_MODEL_DIR=$(value model-dir)
  export MINIMAX_H3_MODEL_DIR
fi

case "$stage" in
  core) value prepare >/dev/null; cmd=(bash build_h3_batch.sh core) ;;
  lm) core=$(value core); cmd=(bash build_h3_batch.sh lm "$core") ;;
  softmax) base=$(value lm); cmd=(bash build_h3_batch.sh softmax "$base") ;;
  generate) runtime=$(value softmax); cmd=(bash run_fl2va_batch.sh --generate-only "$runtime") ;;
  decode) id=$(value generate); cmd=(bash run_fl2va_batch.sh --decode-only "$id") ;;
  compare)
    value decode >/dev/null
    runtime=$(value softmax)
    id=$(value generate)
    cmd=(bash run_fl2va_batch.sh --compare-only "$runtime" "$id")
    ;;
esac

log=$(mktemp "$run/$stage.log.XXXXXX")
printf 'REPRODUCE_LOG=%s\n' "$log"
status=0
"${cmd[@]}" 2>&1 | tee "$log" || status=$?
if (( status != 0 )); then
  [[ $stage == generate ]] && grep -Fq 'OutOfMemoryError' "$log" || exit "$status"
fi

case "$stage" in
  core)
    grep -q '^CORE_IMPORT_OK' "$log"
    output=$(last_value CORE_ARCHIVE)
    test -s "$output"
    save core "${output##*/}"
    ;;
  lm|softmax)
    grep -qx 'RUNTIME_IMPORT_OK' "$log"
    if [[ $stage == softmax ]]; then grep -qx 'H3_SOFTMAX_LONG_INDEX_OK' "$log"; fi
    output=$(last_value RUNTIME_ARCHIVE)
    test -s "$output"
    save "$stage" "${output##*/}"
    ;;
  generate)
    output=$(last_value RESULT_DIR)
    test "$(last_value VAE_INPUT_CHECKPOINT)" = "$output/denoised_latents.pt"
    test -s "$output/denoised_latents.pt"
    save generate "${output##*/}"
    ;;
  decode)
    grep -qx 'VIDEO_FRAMES=124 AUDIO_STREAMS=1' "$log"
    test -s "$(last_value VIDEO_SAVED)"
    save decode OK
    ;;
  compare)
    if grep -qx 'REFERENCE_RESUBMIT_SAME_ARGS' "$log"; then
      echo "Submit the same compare job again with run ID $run_id."
      exit 0
    fi
    grep -qx 'FL2VA_COMPARE_OK' "$log"
    output=$(last_value RESULT_DIR)/compare_fl2va.log
    test -s "$output"
    save compare "$output"
    ;;
esac
printf 'STAGE_COMPLETE=%s\n' "$stage"
cat "$run/$stage"
