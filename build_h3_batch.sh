#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: bash build_h3_batch.sh [core | --configure-only | lm core-ID.tar.gz | softmax runtime-ID.tar.gz]\n' >&2
  exit 2
}

stage=${1:-core}
case "$stage" in
  core|--configure-only) [[ $# -le 1 ]] || usage ;;
  lm) [[ $# == 2 && $2 =~ ^core-[a-zA-Z0-9]+\.tar\.gz$ ]] || usage ;;
  softmax) [[ $# == 2 && $2 =~ ^runtime-[a-zA-Z0-9]+\.tar\.gz$ ]] || usage ;;
  *) usage ;;
esac

: "${IFAB_AFS_HOME:?Run inside an InfiniFabric compute workspace.}"
bundle="$IFAB_AFS_HOME/h3-offline-bootstrap"
scratch=$(mktemp -d "$PWD/minimax-h3-build.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
log_dir="$IFAB_AFS_HOME/minimax-h3-work/logs"
artifact_dir="$IFAB_AFS_HOME/minimax-h3-work/artifacts"
mkdir -p "$log_dir" "$artifact_dir"
run_id=${scratch##*.}
log="$log_dir/build-$run_id.log"
printf 'LOG=%s\n' "$log"

{
  printf 'BUILD_DIR=%s\n' "$scratch"
  printf 'BUILD_SCRIPT=20261113-v5\nBUILD_STAGE=%s\n' "$stage"
  mkdir "$scratch/tmp"
  export TMPDIR="$scratch/tmp"
  export DTK_ROOT=/opt/dtk
  export CUDA_HOME="$DTK_ROOT/cuda/cuda"
  export HYGON_CUDA_HOME="$CUDA_HOME"
  export HYGON_ARCH=gfx936
  export INFINI_ROOT="$scratch/install"
  export PYTHON="$(command -v python)"
  export GIT_CONFIG_NOSYSTEM=1
  export PATH="$scratch/bin:$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$DTK_ROOT/lib:$DTK_ROOT/hip/lib:$CUDA_HOME/lib64:/opt/hyhal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CPLUS_INCLUDE_PATH="$DTK_ROOT/hip/include:$DTK_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

  test -x "$CUDA_HOME/bin/nvcc"
  "$CUDA_HOME/bin/nvcc" --version
  INFINILM_CXX11_ABI=$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
  export INFINILM_CXX11_ABI
  torch_lib=$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent / "lib")')
  export LD_LIBRARY_PATH="$torch_lib:$LD_LIBRARY_PATH"
  printf 'TORCH_ABI=%s\n' "$INFINILM_CXX11_ABI"
  python - "$HYGON_ARCH" <<'PY_DEVICE_CHECK'
import sys

import torch

expected_arch = sys.argv[1]
count = torch.cuda.device_count()
if not torch.version.hip or count < 1:
    raise SystemExit(f'Expected a HIP device for {expected_arch}; found {count} visible devices')
arch = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '')
print(f'PyTorch={torch.__version__} HIP={torch.version.hip} DCU_COUNT={count} DCU_ARCH={arch}')
if arch.split(':', 1)[0] != expected_arch:
    raise SystemExit(f'Expected {expected_arch} on device 0; found {arch or "unknown"}')
PY_DEVICE_CHECK

  mkdir "$scratch/bin"
  install -m 755 "$bundle/xmake-bundle-v3.0.5.linux.x86_64" "$scratch/bin/xmake"
  core="$scratch/InfiniCore"
  lm="$scratch/InfiniLM-minimax_h3-full"
  if [[ $stage == lm ]]; then
    printf 'RESTORING_CORE=%s\n' "$artifact_dir/$2"
    tar -xzf "$artifact_dir/$2" -C "$scratch"
    tar -xzf "$bundle/InfiniLM-minimax_h3-full.tar.gz" -C "$scratch"
    build_file="$lm/xmake.lua"
  else
    tar -xzf "$bundle/InfiniCore-f32e2fbb-offline-source-20261029.tar.gz" -C "$scratch"
    if [[ $stage == softmax ]]; then
      printf 'RESTORING_RUNTIME=%s\n' "$artifact_dir/$2"
      tar -xzf "$artifact_dir/$2" -C "$scratch"
    else
      mkdir "$scratch/wheels" "$scratch/python-deps"
      tar -xzf "$bundle/infinilm-runtime-wheels-py311-x86_64.tar.gz" -C "$scratch/wheels"
      cp "$bundle/"*.whl "$scratch/wheels/"
      for wheel in "$scratch/wheels/"*.whl; do
        python -m zipfile -e "$wheel" "$scratch/python-deps"
      done
    fi
    git -C "$core" rev-parse HEAD
    build_file="$core/xmake.lua"
  fi

  export PYTHONPATH="$core/python:$lm/python:$scratch/python-deps${PYTHONPATH:+:$PYTHONPATH}"
  PYBIND11_INCLUDE=$(python -c 'import pybind11; print(pybind11.get_include())')
  PYTHON_INCLUDE=$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')
  export PYBIND11_INCLUDE PYTHON_INCLUDE
  test -r "$PYBIND11_INCLUDE/pybind11/pybind11.h"
  test -r "$PYTHON_INCLUDE/Python.h"

  python - "$build_file" "$stage" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
requirement = 'add_requires("pybind11")'
package = 'add_packages("pybind11")'
if source.count(requirement) != 1 or package not in source:
    raise SystemExit(f"Unexpected xmake dependency declarations: {path}")
source = source.replace(requirement, '')
source = source.replace(package, 'add_includedirs(os.getenv("PYBIND11_INCLUDE"), os.getenv("PYTHON_INCLUDE"))')
if path.parent.name == 'InfiniCore':
    boost = 'add_requires("boost", {configs = {stacktrace = true}})'
    if source.count(boost) != 1:
        raise SystemExit(f"Unexpected Boost requirement: {path}")
    source = source.replace(boost, 'add_defines("_GLIBCXX_USE_CXX11_ABI=" .. os.getenv("INFINILM_CXX11_ABI"))')
    hygon = path.parent / 'xmake' / 'hygon.lua'
    hygon_source = hygon.read_text()
    target = 'target("infiniop-hygon")\n'
    if hygon_source.count(target) != 1:
        raise SystemExit(f"Unexpected Hygon target declaration: {hygon}")
    serial_cpu_flags = '''    if not has_config("omp") then
        add_cxxflags("-Wno-unknown-pragmas", {force = true})
    end
'''
    hygon.write_text(hygon_source.replace(target, target + serial_cpu_flags))
path.write_text(source)

if sys.argv[2] == 'softmax':
    kernel = path.parent / 'src/infiniop/ops/softmax/cuda/kernel.cuh'
    source = kernel.read_text()
    changes = (
        ('int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize;',
         'const size_t row = blockIdx.x;\n'
         '    const size_t tid = row % stride + (row - row % stride) * dimsize;', 1),
        ('int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;',
         'const size_t otherIdx = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;', 1),
        ('int tid = otherIdx % stride', 'const size_t tid = otherIdx % stride', 1),
        ('for (int ind = threadIdx.x; ind < dimsize;',
         'for (size_t ind = threadIdx.x; ind < dimsize;', 2),
    )
    for before, after, expected_count in changes:
        if source.count(before) != expected_count:
            raise SystemExit(f'Unexpected softmax kernel source: {kernel}: {before}')
        source = source.replace(before, after)
    kernel.write_text(source)
PY

  jobs=${H3_BUILD_JOBS:-6}
  if [[ $stage != lm ]]; then
    if [[ $stage != softmax ]]; then
      cat > "$scratch/mha_hygon_functional_v3.patch" <<'H3_MHA_PATCH'
--- a/src/infinicore/ops/multi_head_attention/mha.cc
+++ b/src/infinicore/ops/multi_head_attention/mha.cc
@@ -1,7 +1,99 @@
 #include "infinicore/ops/mha.hpp"
 #include "../../utils.hpp"
+#include "infinicore/ops/causal_softmax.hpp"
+#include "infinicore/ops/matmul.hpp"
+#include "infinicore/ops/softmax.hpp"
+
+#include <stdexcept>
 
 namespace infinicore::op {
+
+#if defined(ENABLE_HYGON_API) && !defined(ENABLE_FLASH_ATTN)
+namespace {
+
+Tensor expand_kv_heads(const Tensor &input, size_t num_query_heads) {
+    const size_t num_kv_heads = input->size(2);
+    if (num_kv_heads == num_query_heads) {
+        return input;
+    }
+    if (num_kv_heads == 0 || num_query_heads % num_kv_heads != 0) {
+        throw std::runtime_error(
+            "Hygon MHA requires the query head count to be divisible by the KV head count");
+    }
+    const size_t group_size = num_query_heads / num_kv_heads;
+    const Shape expanded_shape{
+        input->size(0), input->size(1), num_kv_heads, group_size, input->size(3)};
+    const Strides expanded_strides{
+        input->stride(0), input->stride(1), input->stride(2), 0, input->stride(3)};
+    return input->as_strided(expanded_shape, expanded_strides)
+        ->contiguous()
+        ->view({input->size(0), input->size(1), num_query_heads, input->size(3)});
+}
+
+Tensor mha_hygon_functional(const Tensor &q,
+                            const Tensor &k,
+                            const Tensor &v,
+                            std::optional<Tensor> alibi_slopes,
+                            float scale,
+                            bool is_causal) {
+    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, v);
+    if (q->ndim() != 4 || k->ndim() != 4 || v->ndim() != 4) {
+        throw std::runtime_error("Hygon MHA expects Q, K and V to be 4-D tensors");
+    }
+    if (q->dtype() != k->dtype() || q->dtype() != v->dtype()) {
+        throw std::runtime_error("Hygon MHA requires matching Q, K and V dtypes");
+    }
+    if (q->dtype() != DataType::F16 && q->dtype() != DataType::BF16
+        && q->dtype() != DataType::F32) {
+        throw std::runtime_error("Hygon functional MHA expects F16, BF16 or F32");
+    }
+    if (alibi_slopes.has_value()) {
+        throw std::runtime_error("Hygon functional MHA does not support ALiBi yet");
+    }
+    for (size_t axis = 0; axis < 4; ++axis) {
+        if (q->size(axis) == 0 || k->size(axis) == 0 || v->size(axis) == 0) {
+            throw std::runtime_error("Hygon functional MHA requires nonempty dimensions");
+        }
+    }
+    if (q->size(0) != k->size(0) || q->size(0) != v->size(0)
+        || k->size(1) != v->size(1)
+        || q->size(3) != k->size(3) || k->size(3) != v->size(3)
+        || k->size(2) != v->size(2)) {
+        throw std::runtime_error("Hygon MHA received incompatible Q, K and V shapes");
+    }
+    if (is_causal && q->size(1) > k->size(1)) {
+        throw std::runtime_error("Hygon causal MHA requires query length <= key length");
+    }
+
+    const size_t batch_size = q->size(0);
+    const size_t query_length = q->size(1);
+    const size_t num_query_heads = q->size(2);
+    const size_t head_size = q->size(3);
+    const size_t key_length = k->size(1);
+    const size_t gemm_batch = batch_size * num_query_heads;
+    auto expanded_k = expand_kv_heads(k, num_query_heads);
+    auto expanded_v = expand_kv_heads(v, num_query_heads);
+
+    // GEMM accepts 2-D/3-D tensors: merge batch and attention heads.
+    auto query = q->permute({0, 2, 1, 3})
+                     ->contiguous()
+                     ->view({gemm_batch, query_length, head_size});
+    auto key = expanded_k->permute({0, 2, 3, 1})
+                   ->contiguous()
+                   ->view({gemm_batch, head_size, key_length});
+    auto value = expanded_v->permute({0, 2, 1, 3})
+                     ->contiguous()
+                     ->view({gemm_batch, key_length, head_size});
+    auto scores = matmul(query, key, scale);
+    auto probabilities = is_causal ? causal_softmax(scores) : softmax(scores, -1);
+    auto output = matmul(probabilities, value, 1.0f);
+    return output->view({batch_size, num_query_heads, query_length, head_size})
+        ->permute({0, 2, 1, 3})
+        ->contiguous();
+}
+
+} // namespace
+#endif
 
 INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MultiheadAttention);
 
@@ -36,6 +128,11 @@
     std::optional<Tensor> alibi_slopes,
     float scale,
     bool is_causal) {
+#if defined(ENABLE_HYGON_API) && !defined(ENABLE_FLASH_ATTN)
+    if (q->device().getType() == Device::Type::HYGON) {
+        return mha_hygon_functional(q, k, v, alibi_slopes, scale, is_causal);
+    }
+#endif
     auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
     mha_(out, q, k, v, alibi_slopes, scale, is_causal);
     return out;
@@ -48,6 +145,17 @@
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
+    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v);
+#if defined(ENABLE_HYGON_API) && !defined(ENABLE_FLASH_ATTN)
+    if (out->device().getType() == Device::Type::HYGON) {
+        if (out->shape() != q->shape() || out->dtype() != q->dtype()) {
+            throw std::runtime_error("Hygon MHA output must match query shape and dtype");
+        }
+        auto result = mha_hygon_functional(q, k, v, alibi_slopes, scale, is_causal);
+        out->copy_from(result);
+        return;
+    }
+#endif
     MultiheadAttention::execute(out, q, k, v, alibi_slopes, scale, is_causal);
 }
 
H3_MHA_PATCH
      git -C "$core" apply --check "$scratch/mha_hygon_functional_v3.patch"
      git -C "$core" apply "$scratch/mha_hygon_functional_v3.patch"
    fi

    cd "$core"
    printf '\nConfiguring InfiniCore for gfx936 with ATen and CCL\n'
    xmake f -y -m release --hygon-dcu=y --hygon-arch=gfx936 \
      --cuda="$CUDA_HOME" --aten=y --ccl=y --omp=n --policies=network.mode:private
    printf 'INFINICORE_CONFIGURE_OK\n'

    if [[ $stage == --configure-only ]]; then
      exit 0
    fi
  fi

  if [[ $stage == core ]]; then
    xmake build -y -j "$jobs" --linkjobs=1 infinicore_cpp_api
    for target in infinirt infiniop infiniccl infinicore_cpp_api; do
      xmake install "$target"
    done
    xmake build -y -j "$jobs" --linkjobs=1 _infinicore
    xmake install _infinicore

    archive="$artifact_dir/core-$run_id.tar.gz"
    tar -cf - -C "$scratch" install python-deps InfiniCore/python | gzip -1 > "$archive.partial"
    mv "$archive.partial" "$archive"
    printf 'CORE_ARCHIVE=%s\n' "$archive"
    printf 'NEXT_ARGS=lm %s\n' "${archive##*/}"
    python -c 'import torch; from infinicore.lib import _infinicore; print("CORE_IMPORT_OK", _infinicore.__file__)'
    exit 0
  fi

  if [[ $stage == softmax ]]; then
    xmake build -y -j "$jobs" --linkjobs=1 infiniop
    xmake install infiniop
    PYTHONPATH="$core/test/infiniop:$PYTHONPATH" python - <<'PY'
import ctypes

import torch
from libinfiniop import (
    CTensor, InfiniDeviceEnum, InfiniDtype, LIBINFINIOP, check_error,
    create_handle, destroy_handle, infiniopOperatorDescriptor_t,
)

shape = (7, 27570, 27570)
print(f'Checking Hygon BF16 softmax with shape {shape}', flush=True)
torch.cuda.set_device(0)
check_error(LIBINFINIOP.infinirtSetDevice(InfiniDeviceEnum.HYGON, 0))
handle = create_handle()
tensor = CTensor(InfiniDtype.BF16, shape, None)
descriptor = infiniopOperatorDescriptor_t()
check_error(LIBINFINIOP.infiniopCreateSoftmaxDescriptor(
    handle, ctypes.byref(descriptor), tensor.descriptor, tensor.descriptor, -1,
))
try:
    data = torch.zeros(shape, device='cuda:0', dtype=torch.bfloat16)
    output = torch.empty_like(data)
    data[-1, -1, -1] = 2
    torch.cuda.synchronize()
    workspace_size = ctypes.c_uint64()
    check_error(LIBINFINIOP.infiniopGetSoftmaxWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = torch.empty(workspace_size.value, device='cuda:0', dtype=torch.uint8)
    check_error(LIBINFINIOP.infiniopSoftmax(
        descriptor, workspace.data_ptr(), workspace_size.value,
        output.data_ptr(), data.data_ptr(), torch.cuda.current_stream().cuda_stream,
    ))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0, 0], data[0, 0].softmax(-1), atol=0, rtol=0.01)
    torch.testing.assert_close(output[-1, -1], data[-1, -1].softmax(-1), atol=0, rtol=0.01)
    print('H3_SOFTMAX_LONG_INDEX_OK', flush=True)
finally:
    check_error(LIBINFINIOP.infiniopDestroySoftmaxDescriptor(descriptor))
    tensor.destroy_desc()
    destroy_handle(handle)
PY
  else
    cd "$lm"
    xmake f -y -m release --cxx11-abi="$INFINILM_CXX11_ABI" --policies=network.mode:private
    xmake build -y -j "$jobs" --linkjobs=1 _infinilm
    xmake install _infinilm
  fi
  archive="$artifact_dir/runtime-$run_id.tar.gz"
  tar -cf - -C "$scratch" install python-deps InfiniCore/python \
    InfiniLM-minimax_h3-full/python InfiniLM-minimax_h3-full/examples | gzip -1 > "$archive.partial"
  mv "$archive.partial" "$archive"
  printf 'RUNTIME_ARCHIVE=%s\n' "$archive"

  python - <<'PY'
import importlib.util
import torch
import infinicore
import infinilm
from infinicore.lib import _infinicore
from infinilm.lib import _infinilm

print(f"InfiniCore: {_infinicore.__file__}")
print(f"InfiniLM: {_infinilm.__file__}")
print("RUNTIME_IMPORT_OK")
for module in ("diffusers", "transformers", "accelerate", "av", "skimage"):
    print(f"{module}: {'found' if importlib.util.find_spec(module) else 'missing'}")
PY
} 2>&1 | tee "$log"
