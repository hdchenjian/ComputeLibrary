#ifndef __ARM_COMPUTE_CLPRELULAYER_H__
#define __ARM_COMPUTE_CLPRELULAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLPreluLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLPreluLayerKernel */
class CLPreluLayer : public IFunction
{
public:
    /** Default constructor */
    CLPreluLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPreluLayer(const CLPreluLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPreluLayer &operator=(const CLPreluLayer &) = delete;
    /** Allow instances of this class to be moved */
    CLPreluLayer(CLPreluLayer &&) = default;
    /** Allow instances of this class to be moved */
    CLPreluLayer &operator=(CLPreluLayer &&) = default;
    /** Default destructor */
    virtual ~CLPreluLayer() = default;

    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in]  input             Source tensor. Data type supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output            Destination tensor. Data types supported: same as @p input.
     * @param[in]  info              Contains stride information described in @ref Size2D.
     * @param[in]  upsampling_policy Defines the policy to fill the intermediate pixels.
     */
    void configure(ICLTensor *input, ICLTensor *output, ICLTensor *slope);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayerPrelu
     *
     * @param[in] input             Source tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] output            Destination tensor info. Data types supported: same as @p input.
     * @param[in] info              Contains  stride information described in @ref Size2D.
     * @param[in] upsampling_policy Defines the policy to fill the intermediate pixels.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    CLPreluLayerKernel _prelu;
    ICLTensor            *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLPRELULAYER_H__ */
