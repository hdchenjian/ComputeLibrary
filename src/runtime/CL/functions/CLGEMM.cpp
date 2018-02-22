/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/CL/functions/CLGEMM.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

using namespace arm_compute;

namespace
{
inline bool is_interleaved_transposed(int m, int n, int k, DataType data_type, bool reshape_b_only_on_first_run, GPUTarget gpu_target)
{
    bool flag = true;

    if(gpu_target == GPUTarget::BIFROST)
    {
        if(k > 256 && m > 4 && data_type == DataType::F32 && reshape_b_only_on_first_run)
        {
            const float scale = k < 1024 ? 2.0f : 2.5f;
            flag              = (scale * n) > ((1.66f * n) + 38.4f);
        }
        else
        {
            flag = false;
        }
    }

    return flag;
}

Status validate_arguments(const ITensorInfo *a, const ITensorInfo *b, const ICLTensor *c, const ITensorInfo *output, const float alpha, const float beta, const GEMMInfo &gemm_info = GEMMInfo())
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    if(c != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, c->info());
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(1) != c->info()->dimension(1), "The C matrix must have the same number of rows as the matrix A");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(b->dimension(0) != c->info()->dimension(0), "The C matrix must have the same number of columns as the matrix B");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(c->info()->dimension(0) != output->dimension(0), "The C matrix must have the same number of rows as the output matrix");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(c->info()->dimension(1) != output->dimension(1), "The C matrix must have the same number of columns as the output matrix");
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(0) != b->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");

    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    return Status{};
}
} // namespace

CLGEMM::CLGEMM(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _interleave_kernel(), _transpose_kernel(), _mm_kernel(), _ma_kernel(), _tmp_a(), _tmp_b(), _is_interleaved_transposed(false), _run_addition(false),
      _is_first_run(true), _reshape_b_only_on_first_run(false)
{
}

void CLGEMM::configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(a->info(), b->info(), c, output->info(), alpha, beta, gemm_info));

    // Check if we need to reshape the matrix B only on the first run
    _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();

    const ICLTensor *matrix_a = a;
    const ICLTensor *matrix_b = b;

    // Get the GPU target
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Set the target for the kernels
    _interleave_kernel.set_target(gpu_target);
    _mm_kernel.set_target(gpu_target);

    // Arguments used by GEMMReshapeInfo
    // If we pass the matrix A and matrix B reshaped to CLGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to CLGEMMReshapeInfo
    // in order to know how the matrices have been reshaped
    const int m                         = a->info()->dimension(1);
    const int n                         = b->info()->dimension(0);
    const int k                         = a->info()->dimension(0);
    int       mult_transpose1xW_width   = 1;
    int       mult_interleave4x4_height = 1;

    if(gpu_target == GPUTarget::BIFROST)
    {
        mult_transpose1xW_width   = 4;
        mult_interleave4x4_height = 2;
    }

    // Check if we need to reshape the matrix A and matrix B
    _is_interleaved_transposed = is_interleaved_transposed(m, n, k, a->info()->data_type(), _reshape_b_only_on_first_run, gpu_target);

    if(_is_interleaved_transposed)
    {
        matrix_a = &_tmp_a;
        matrix_b = &_tmp_b;

        // Manage intermediate buffers
        _memory_group.manage(&_tmp_a);
        _memory_group.manage(&_tmp_b);

        // _tmp_a and _tmp_b will be auto configured in _interleave_kernel and in _transpose_kernel

        // Configure interleave kernel
        _interleave_kernel.configure(a, &_tmp_a, mult_interleave4x4_height);

        // Configure transpose kernel
        _transpose_kernel.configure(b, &_tmp_b, mult_transpose1xW_width);
    }

    _mm_kernel.configure(matrix_a, matrix_b, output, alpha, _is_interleaved_transposed, GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height));

    if(_is_interleaved_transposed)
    {
        // Allocate intermediate tensors
        _tmp_a.allocator()->allocate();
        _tmp_b.allocator()->allocate();
    }

    // Configure matrix addition kernel
    if(beta != 0 && c != nullptr)
    {
        _ma_kernel.configure(c, output, beta);
        _run_addition = true;
    }
}

Status CLGEMM::validate(const ITensorInfo *a, const ITensorInfo *b, const ICLTensor *c, const ITensorInfo *output, const float alpha, const float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(a, b, c, output, alpha, beta, gemm_info));
    return Status{};
}

void CLGEMM::run()
{
    _memory_group.acquire();

    if(_is_interleaved_transposed)
    {
        // Run interleave kernel
        CLScheduler::get().enqueue(_interleave_kernel, false);

        if(_is_first_run)
        {
            // Run transpose kernel
            CLScheduler::get().enqueue(_transpose_kernel, false);

            _is_first_run = false;
        }
        else if(!_reshape_b_only_on_first_run)
        {
            // Run transpose kernel
            CLScheduler::get().enqueue(_transpose_kernel, false);
        }
    }

    // Run matrix multiply kernel
    CLScheduler::get().enqueue(_mm_kernel, !_run_addition);

    // Run matrix addition kernel
    if(_run_addition)
    {
        CLScheduler::get().enqueue(_ma_kernel);
    }

    _memory_group.release();
}