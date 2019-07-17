#ifndef __ARM_COMPUTE_NEPRELULAYER_H__
#define __ARM_COMPUTE_NEPRELULAYER_H__

#include "arm_compute/core/NEON/kernels/NEPreluLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Function to run prelu layer */
class NEPreluLayer : public IFunction
{
public:
    /** Constructor */
    NEPreluLayer();
    /** Set the input output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/F16/F32.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     * @param[in]  info   Contains stride information described in @ref Size2D.
     * @param[in]  policy Defines the policy to fill the intermediate pixels.
     *
     */
    void configure(const ITensor *input, ITensor *output, ITensor *slope);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPreluLayer
     *
     * @param[in]  input  Source tensor info. Data types supported: QASYMM8/F16/F32.
     * @param[out] output Destination tensor info. Data types supported: same as @p input.
     * @param[in]  info   Contains stride information described in @ref Size2D.
     * @param[in]  policy Defines the policy to fill the intermediate pixels.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    NEPreluLayerKernel _kernel;
    DataLayout            _data_layout;
};
} // arm_compute
#endif /* __ARM_COMPUTE_NEPRELULAYER_H__ */
