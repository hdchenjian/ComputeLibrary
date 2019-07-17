#include "arm_compute/runtime/CL/functions/CLPreluLayer.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
CLPreluLayer::CLPreluLayer() // NOLINT
    : _prelu(),
      _output(nullptr)
{
}

Status CLPreluLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLPreluLayerKernel::validate(input, output);
}

void CLPreluLayer::configure(ICLTensor *input, ICLTensor *output, ICLTensor *slope)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _output = output;
    _prelu.configure(input, _output, slope);
}

void CLPreluLayer::run()
{
    CLScheduler::get().enqueue(_prelu, false);
}
} // namespace arm_compute
