#include "arm_compute/runtime/NEON/functions/NEPreluLayer.h"

#include "arm_compute/core/NEON/kernels/NEPreluLayerKernel.h"

namespace arm_compute
{
NEPreluLayer::NEPreluLayer()
    : _kernel(), _data_layout()
{
}

Status NEPreluLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NEPreluLayerKernel::validate(input, output);
}

void NEPreluLayer::configure(const ITensor *input, ITensor *output, ITensor *slope)
{
    _data_layout = input->info()->data_layout();
    _kernel.configure(input, output, slope);
}

void NEPreluLayer::run()
{
    const auto win = (_data_layout == DataLayout::NCHW) ? Window::DimZ : Window::DimX;
    NEScheduler::get().schedule(&_kernel, win);
}
} // namespace arm_compute
