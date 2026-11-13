#!/usr/bin/env bash
set -euo pipefail

batch_started=$SECONDS
mode=${1:-}
if [[ $mode == --check-only || $mode == --generate-only || $mode == --decode-only || $mode == --compare-only ]]; then
  shift
else
  mode=
fi
usage() {
  printf 'Usage: bash run_fl2va_batch.sh [--check-only | --generate-only] runtime-ID.tar.gz\n' >&2
  printf '       bash run_fl2va_batch.sh --decode-only fl2va-ID\n' >&2
  printf '       bash run_fl2va_batch.sh --compare-only runtime-ID.tar.gz fl2va-ID [reference.mp4]\n' >&2
  exit 2
}
case "$mode" in
  --decode-only) [[ $# == 1 && $1 =~ ^fl2va-[a-zA-Z0-9]+$ ]] || usage ;;
  --compare-only)
    [[ $# -ge 2 && $# -le 3 && $1 =~ ^runtime-[a-zA-Z0-9]+\.tar\.gz$ && $2 =~ ^fl2va-[a-zA-Z0-9]+$ ]] || usage
    [[ $# == 2 || $3 =~ ^[a-zA-Z0-9_.-]+\.mp4$ ]] || usage
    ;;
  *) [[ $# == 1 && $1 =~ ^runtime-[a-zA-Z0-9]+\.tar\.gz$ ]] || usage ;;
esac

: "${IFAB_AFS_HOME:?Run inside an InfiniFabric compute workspace.}"
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
afs_work="$IFAB_AFS_HOME/minimax-h3-work"
if [[ $mode == --decode-only ]]; then
  result_dir="$afs_work/results/$1"
  run_input="$result_dir/denoised_latents.pt"
else
  run_input="$afs_work/artifacts/$1"
fi
if [[ $mode == --compare-only ]]; then
  result_dir="$afs_work/results/$2"
  test -s "$result_dir/infinilm_fl2va.mp4" || { printf 'Missing InfiniLM video: %s\n' "$result_dir" >&2; exit 1; }
  reference="${3:+$IFAB_AFS_HOME/$3}"
  if [[ -n $reference && ! -s $reference ]]; then
    printf 'Missing reference video: %s\n' "$reference" >&2
    exit 1
  fi
fi
deps_archive_one="$IFAB_AFS_HOME/h3-fl2va-wheels-part1.tar.gz"
deps_archive_two="$IFAB_AFS_HOME/h3-fl2va-wheels-part2.tar.gz"
compat_archive="$IFAB_AFS_HOME/h3-fl2va-compat.tar.gz"
model_dir=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers

for path in "$run_input" "$deps_archive_one" "$deps_archive_two" "$compat_archive" "$model_dir/model_index.json"; do
  if [[ ! -r $path ]]; then
    printf 'Missing or unreadable: %s\n' "$path" >&2
    exit 1
  fi
done

scratch=$(mktemp -d "$PWD/minimax-h3-run.XXXXXX")
mkdir "$scratch/deps" "$scratch/tmp"
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch/tmp"
export GIT_CONFIG_NOSYSTEM=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

tar -xzf "$deps_archive_one" -C "$scratch"
tar -xzf "$deps_archive_two" -C "$scratch"
tar -xzf "$compat_archive" -C "$scratch"
deps_dir="$scratch/h3-fl2va-wheels"
(cd "$deps_dir" && sha256sum -c SHA256SUMS.part1 && sha256sum -c SHA256SUMS.part2 && sha256sum -c SHA256SUMS.compat)
if [[ $mode != --decode-only ]]; then
  tar -xzf "$run_input" -C "$scratch"
fi
for wheel in "$deps_dir/"*.whl; do
  if [[ ${wheel##*/} == numpy-2.* ]]; then
    continue
  fi
  python -m zipfile -e "$wheel" "$scratch/deps"
done

python - "$scratch/deps/diffusers" <<'PY'
from pathlib import Path
import sys

diffusers = Path(sys.argv[1])
video_vae = diffusers / 'models/autoencoders/autoencoder_kl_minimax_h3.py'
source = video_vae.read_text()
backend = 'backend=self._attention_backend,'
if source.count(backend) != 1:
    raise SystemExit(f'Unexpected MiniMax-H3 VAE attention code: {video_vae}')
video_vae.write_text(source.replace(backend, 'backend=self._attention_backend or "_native_math",'))

decoders = diffusers / 'modular_pipelines/minimax_h3/decoders.py'
source = decoders.read_text()
unpack_audio = '        block_state.audio_latents = audio_rows.permute(0, 2, 1).contiguous()\n'
if source.count(unpack_audio) != 1:
    raise SystemExit(f'Unexpected MiniMax-H3 latent unpacking code: {decoders}')
checkpoint = '''
        checkpoint_path = os.environ.get("H3_LATENT_CHECKPOINT")
        if checkpoint_path:
            partial = checkpoint_path + ".partial"
            torch.save({
                "latents": block_state.latents.detach().cpu(),
                "audio_latents": block_state.audio_latents.detach().cpu(),
            }, partial)
            os.replace(partial, checkpoint_path)
            print(f"VAE_INPUT_CHECKPOINT={checkpoint_path}", flush=True)
'''
source = source.replace('import torch\n', 'import os\n\nimport torch\n', 1)
decoders.write_text(source.replace(unpack_audio, unpack_audio + checkpoint))
PY

export DTK_ROOT=/opt/dtk
export CUDA_HOME="$DTK_ROOT/cuda/cuda"
export INFINI_ROOT="$scratch/install"
torch_lib=$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("torch").origin).parent / "lib")')
export LD_LIBRARY_PATH="$torch_lib:$INFINI_ROOT/lib:$DTK_ROOT/lib:$DTK_ROOT/hip/lib:$CUDA_HOME/lib64:/opt/hyhal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$scratch/deps:$scratch/python-deps:$scratch/InfiniCore/python:$scratch/InfiniLM-minimax_h3-full/python${PYTHONPATH:+:$PYTHONPATH}"

if [[ $mode == --decode-only ]]; then
  printf 'RESULT_DIR=%s\n' "$result_dir"
  python - "$model_dir" "$run_input" "$result_dir" <<'PY' 2>&1 | tee "$result_dir/decode_fl2va.log"
from pathlib import Path
import sys
import time

import av
import torch
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3DecodeStep
from diffusers.utils.export_utils import encode_video

model_dir, checkpoint, output_dir = map(Path, sys.argv[1:])
output = output_dir / 'infinilm_fl2va.mp4'
partial = output_dir / '.infinilm_fl2va.partial.mp4'
device = torch.device('cuda:0')
torch.cuda.set_device(device)
free, total = torch.cuda.mem_get_info(device)
print(f'DECODE_VRAM_GIB={free / 2**30:.2f}/{total / 2**30:.2f}', flush=True)

latents = torch.load(checkpoint, map_location='cpu', weights_only=True)
if set(latents) != {'latents', 'audio_latents'}:
    raise ValueError(f'Unexpected latent fields in {checkpoint}')
print(f'RESTORED_VAE_INPUT={checkpoint}', flush=True)

started = time.perf_counter()
pipe = MiniMaxH3DecodeStep().init_pipeline(str(model_dir))
pipe.load_components(
    names=['vae', 'audio_vae'], dtype=torch.bfloat16,
    pretrained_model_name_or_path=str(model_dir), local_files_only=True,
)
if pipe.vae is None or pipe.audio_vae is None:
    raise RuntimeError('Could not load both MiniMax-H3 VAE components')
for vae in (pipe.vae, pipe.audio_vae):
    vae.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

with torch.inference_mode():
    result = pipe(
        latents=latents['latents'].to(device),
        audio_latents=latents['audio_latents'].to(device),
        output_type='pil', output=['videos', 'audio', 'sampling_rate'],
    )
    encode_video(
        result['videos'][0], fps=pipe.fps, output_path=str(partial),
        audio=result['audio'][0], audio_sample_rate=result['sampling_rate'],
    )

with av.open(str(partial)) as container:
    frame_count = sum(1 for _ in container.decode(video=0))
    audio_streams = len(container.streams.audio)
if frame_count != 124 or audio_streams != 1:
    raise RuntimeError(f'Expected 124 video frames and one audio stream, got {frame_count} and {audio_streams}')
partial.replace(output)
print(f'VIDEO_FRAMES={frame_count} AUDIO_STREAMS={audio_streams}')
print(f'DECODE_AND_EXPORT_SECONDS={time.perf_counter() - started:.3f}')
print(f'VIDEO_SAVED={output}')
PY
  exit 0
fi

if [[ $mode == --compare-only ]]; then
  examples="$scratch/InfiniLM-minimax_h3-full/examples"
  compare_args=(--tasks=fl2va --model-path="$model_dir" --first-frame="$project_dir/first_frame.png"
    --output-dir="$result_dir" --infinilm-fl2va-video="$result_dir/infinilm_fl2va.mp4"
    --legacy-gpus=0,1,2,3,4,5,6,7 --legacy-memory-fraction=0.4)
  if [[ -n $reference ]]; then
    compare_args+=(--reference-fl2va-video="$reference")
  else
    reference="$result_dir/reference_fl2va.mp4"
    if [[ ! -s $reference ]]; then
      python - "$scratch/deps/diffusers" "$examples/minimax_h3.py" <<'PY'
from pathlib import Path
import sys

transformer = Path(sys.argv[1]) / 'models/transformers/transformer_minimax_h3.py'
source = transformer.read_text()
start = '        hidden_states = dispatch_attention_fn(\n'
end = '        hidden_states = hidden_states.flatten(2, 3).type_as(query)\n'
if source.count(start) != 1 or source.count(end) != 1:
    raise SystemExit(f'Unexpected MiniMax-H3 transformer attention code: {transformer}')
begin, tail = source.split(start, 1)
_, finish = tail.split(end, 1)
attention = '''        if attention_mask is not None or self._parallel_config is not None:
            raise RuntimeError("This FL2VA reference expects unmasked attention on each model device")
        hidden_states = torch.cat(
            [dispatch_attention_fn(query_part, key, value, backend="_native_math")
             for query_part in query.split(512, dim=1)],
            dim=1,
        )
'''
transformer.write_text(begin + attention + end + finish)

generator = Path(sys.argv[2])
source = generator.read_bytes()
anchor = b'                subfolder="text_encoder",'
if source.count(anchor) != 1:
    raise SystemExit(f'Unexpected PyTorch text-encoder loading code: {generator}')
newline = b'\r\n' if b'\r\n' in source else b'\n'
generator.write_bytes(source.replace(anchor, anchor + newline + b'                attn_implementation="eager",'))

denoise = Path(sys.argv[1]) / 'modular_pipelines/minimax_h3/denoise.py'
source = denoise.read_text()
loop = '''        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
'''
if source.count(loop) != 1:
    raise SystemExit(f'Unexpected MiniMax-H3 denoise loop: {denoise}')
resume_loop = '''        checkpoint = os.environ.get("H3_REFERENCE_CHECKPOINT")
        first = 0
        total = len(block_state.timesteps)
        if checkpoint and os.path.isfile(checkpoint):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            first = saved["next_step"]
            if not 0 <= first <= total:
                raise ValueError(f"Invalid saved reference step: {first}")
            for name in ("timesteps", "audio_timesteps"):
                if not torch.equal(saved[name], getattr(block_state, name).cpu()):
                    raise ValueError(f"Reference schedule changed: {name}")
            for name in ("latents", "audio_latents"):
                target = getattr(block_state, name)
                if target.shape != saved[name].shape or target.dtype != saved[name].dtype:
                    raise ValueError(f"Reference latent layout changed: {name}")
                target.copy_(saved[name])
            if first == total:
                # The downstream blocks consume the final latents. A completed
                # checkpoint has no per-step predictions to publish.
                state.set("latents", block_state.latents)
                state.set("audio_latents", block_state.audio_latents)
                print(f"REFERENCE_READY_FOR_DECODE={first}/{total}", flush=True)
                return components, state
            components.scheduler.set_begin_index(first)
            components.audio_scheduler.set_begin_index(first)
        if checkpoint:
            print(f"REFERENCE_RESUME={first}/{total}", flush=True)
        deadline = float(os.environ.get("H3_REFERENCE_DEADLINE", "inf"))
        with self.progress_bar(total=total - first) as progress_bar:
            for i in range(first, total):
                started = time.monotonic()
                t = block_state.timesteps[i]
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
                if checkpoint:
                    next_step = i + 1
                    saved = {name: getattr(block_state, name).detach().cpu() for name in
                             ("timesteps", "audio_timesteps", "latents", "audio_latents")}
                    saved["next_step"] = next_step
                    partial = f"{checkpoint}.{os.getpid()}.partial"
                    torch.save(saved, partial)
                    os.replace(partial, checkpoint)
                    print(f"REFERENCE_STEP_SAVED={next_step}/{total}", flush=True)
                    next_budget = max(130, (time.monotonic() - started) * 1.3)
                    if time.time() + next_budget > deadline:
                        print(f"REFERENCE_SLICE_COMPLETE={next_step}/{total}", flush=True)
                        raise SystemExit(75)
'''
source = source.replace('import torch\n', 'import os\nimport time\n\nimport torch\n', 1)
denoise.write_text(source.replace(loop, resume_loop))
PY
      printf 'REFERENCE_BACKEND=PyTorch (eager Qwen3-VL, Math SDPA with 512 query rows per chunk for H3)\n'
      printf 'RESULT_DIR=%s\n' "$result_dir"
      deadline=$(( $(date +%s) + 1640 - (SECONDS - batch_started) ))
      reference_partial="$result_dir/.reference_fl2va.partial.mp4"
      status=0
      H3_REFERENCE_CHECKPOINT="$result_dir/reference_steps.pt" \
      H3_REFERENCE_DEADLINE="$deadline" \
      H3_LATENT_CHECKPOINT="$result_dir/reference_denoised_latents.pt" \
        python "$examples/minimax_h3.py" fl2va --legacy --model-path="$model_dir" \
          --first-frame="$project_dir/first_frame.png" --text='A dog falls asleep' \
          --gpus=0,1,2,3,4,5,6,7 --legacy-memory-fraction=0.4 \
          --seed=42 --steps=50 --num-frames=124 --output="$reference_partial" \
          2>&1 | tee -a "$result_dir/reference_fl2va.log" || status=$?
      if [[ $status == 75 ]]; then
        printf 'REFERENCE_RESUBMIT_SAME_ARGS\n'
        exit 0
      fi
      [[ $status == 0 ]] || exit "$status"
      test -s "$reference_partial"
      mv -f "$reference_partial" "$reference"
    fi
    compare_args+=(--reference-fl2va-video="$reference")
  fi
  printf 'RESULT_DIR=%s\n' "$result_dir"
  python "$examples/minimax_h3_compare.py" "${compare_args[@]}" 2>&1 | tee "$result_dir/compare_fl2va.log"
  grep -Eq '^[[:space:]]+Size:[[:space:]]+PASS' "$result_dir/compare_fl2va.log"
  grep -Eq '^[[:space:]]+Length:[[:space:]]+PASS \(PyTorch 124 frames, InfiniLM 124 frames\)' "$result_dir/compare_fl2va.log"
  grep -Eq '^[[:space:]]+Average SSIM:[[:space:]]+-?[0-9]+(\.[0-9]+)?%' "$result_dir/compare_fl2va.log"
  printf 'FL2VA_COMPARE_OK\n'
  exit 0
fi

mkdir -p "$afs_work/results"
run_id=${scratch##*.}
result_dir="$afs_work/results/fl2va-$run_id"
printf 'RESULT_DIR=%s\n' "$result_dir"

python - "$model_dir" "$project_dir/first_frame.png" <<'PY'
import sys

import numpy as np
import torch
import torchvision
from diffusers.models.attention_dispatch import dispatch_attention_fn
from PIL import Image
from transformers import Qwen3VLProcessor

print(f"NumPy {np.__version__}; torchvision {torchvision.__version__}")
values = np.array([1, 2], dtype=np.float32)
if not np.array_equal(torch.from_numpy(values).numpy(), values):
    raise RuntimeError("PyTorch/NumPy round trip failed")
processor = Qwen3VLProcessor.from_pretrained(sys.argv[1], subfolder="processor", local_files_only=True)
with Image.open(sys.argv[2]) as frame:
    vision = processor.image_processor(images=[frame.convert("RGB")], return_tensors="pt")
if vision["pixel_values"].numel() == 0 or vision["image_grid_thw"].numel() == 0:
    raise RuntimeError("Qwen3-VL processor returned empty first-frame features")
print("FL2VA_PROCESSOR_OK")
with torch.inference_mode():
    q = torch.zeros((1, 64, 32, 64), device="cuda:0", dtype=torch.float16)
    v = torch.ones_like(q)
    out = dispatch_attention_fn(q, q, v, backend="_native_math")
    torch.testing.assert_close(out, v)
print("FL2VA_VAE_SDPA_OK backend=_native_math")
PY

run_args=("$scratch/InfiniLM-minimax_h3-full" "$model_dir" "$result_dir")
if [[ -n $mode ]]; then
  run_args=("$mode" "${run_args[@]}")
fi
bash "$project_dir/run_fl2va.sh" "${run_args[@]}"
