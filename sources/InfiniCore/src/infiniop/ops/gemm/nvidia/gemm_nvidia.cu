#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "gemm_nvidia.cuh"
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#include <cublasLt.h>
#endif

namespace op::gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    cublasLtHandle_t lt_handle = nullptr;
    cublasLtMatmulDesc_t lt_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;

    void destroyLtDescriptors();
    bool createBf16LtDescriptors(const MatmulInfo &info);
#endif
};

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
static size_t ltLayoutRows(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.rows : matrix.cols;
}

static size_t ltLayoutCols(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.cols : matrix.rows;
}

static bool setLtLayoutBatch(cublasLtMatrixLayout_t layout, const BlasMatrix &matrix, size_t batch) {
    int32_t batch_count = static_cast<int32_t>(batch);
    int64_t stride = static_cast<int64_t>(matrix.stride);
    return cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
               &batch_count, sizeof(batch_count))
            == CUBLAS_STATUS_SUCCESS
        && cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
               &stride, sizeof(stride))
               == CUBLAS_STATUS_SUCCESS;
}

void Descriptor::Opaque::destroyLtDescriptors() {
    if (a_layout) {
        cublasLtMatrixLayoutDestroy(a_layout);
        a_layout = nullptr;
    }
    if (b_layout) {
        cublasLtMatrixLayoutDestroy(b_layout);
        b_layout = nullptr;
    }
    if (c_layout) {
        cublasLtMatrixLayoutDestroy(c_layout);
        c_layout = nullptr;
    }
    if (lt_desc) {
        cublasLtMatmulDescDestroy(lt_desc);
        lt_desc = nullptr;
    }
    if (lt_handle) {
        cublasLtDestroy(lt_handle);
        lt_handle = nullptr;
    }
}

bool Descriptor::Opaque::createBf16LtDescriptors(const MatmulInfo &info) {
    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    if (cublasLtCreate(&lt_handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescCreate(&lt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSA,
            &op_a, sizeof(op_a))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSB,
            &op_b, sizeof(op_b))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    if (cublasLtMatrixLayoutCreate(
            &a_layout, CUDA_R_16BF,
            ltLayoutRows(info.a_matrix), ltLayoutCols(info.a_matrix),
            info.a_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &b_layout, CUDA_R_16BF,
            ltLayoutRows(info.b_matrix), ltLayoutCols(info.b_matrix),
            info.b_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &c_layout, CUDA_R_16BF,
            ltLayoutRows(info.c_matrix), ltLayoutCols(info.c_matrix),
            info.c_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    return setLtLayoutBatch(a_layout, info.a_matrix, info.batch)
        && setLtLayoutBatch(b_layout, info.b_matrix, info.batch)
        && setLtLayoutBatch(c_layout, info.c_matrix, info.batch);
}
#endif

Descriptor::~Descriptor() {
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    if (_opaque) {
        _opaque->destroyLtDescriptors();
    }
#endif
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    auto info = result.take();
    auto opaque = new Opaque{handle->internal()};
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    if (dtype == INFINI_DTYPE_BF16 && !opaque->createBf16LtDescriptors(info)) {
        opaque->destroyLtDescriptors();
    }
#endif

    *desc_ptr = new Descriptor(
        dtype, info, 0,
        opaque,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    if (_dtype == INFINI_DTYPE_BF16 && _opaque->lt_handle && _opaque->lt_desc
        && _opaque->a_layout && _opaque->b_layout && _opaque->c_layout) {
        auto lt_status = cublasLtMatmul(
            _opaque->lt_handle,
            _opaque->lt_desc,
            &alpha,
            a,
            _opaque->a_layout,
            b,
            _opaque->b_layout,
            &beta,
            c,
            _opaque->c_layout,
            c,
            _opaque->c_layout,
            nullptr,
            workspace,
            workspace_size,
            (cudaStream_t)stream);
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            return INFINI_STATUS_SUCCESS;
        }
    }
#endif

    CHECK_STATUS(_opaque->internal->useCublas(
        (cudaStream_t)stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(
                cublasGemmStridedBatchedEx(
                    handle,
                    op_a,
                    op_b,
                    static_cast<int>(_info.m),
                    static_cast<int>(_info.n),
                    static_cast<int>(_info.k),
                    &alpha,
                    a,
                    a_type,
                    static_cast<int>(_info.a_matrix.ld()),
                    _info.a_matrix.stride,
                    b,
                    b_type,
                    static_cast<int>(_info.b_matrix.ld()),
                    _info.b_matrix.stride,
                    &beta,
                    c,
                    c_type,
                    static_cast<int>(_info.c_matrix.ld()),
                    _info.c_matrix.stride,
                    static_cast<int>(_info.batch),
                    compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::nvidia
