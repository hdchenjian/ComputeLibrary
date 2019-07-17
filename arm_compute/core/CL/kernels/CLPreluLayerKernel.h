#ifndef __ARM_COMPUTE_CLPRELULAYERKERNEL_H__
#define __ARM_COMPUTE_CLPRELULAYERKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the PreluLayer kernel on OpenCL. */
class CLPreluLayerKernel : public ICLKernel
{
public:
    /** Constructor */
    CLPreluLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPreluLayerKernel(const CLPreluLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPreluLayerKernel &operator=(const CLPreluLayerKernel &) = delete;
    /** Default Move Constructor. */
    CLPreluLayerKernel(CLPreluLayerKernel &&) = default;
    /** Default move assignment operator */
    CLPreluLayerKernel &operator=(CLPreluLayerKernel &&) = default;
    /** Default destructor */
    ~CLPreluLayerKernel() = default;

    /** Initialise the kernel's input and output.
     *
     * @param[in]  input             Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output            Destination tensor. Data types supported: same as @p input.
     * @param[in]  info              Contains stride information described in @ref Size2D.
     * @param[in]  upsampling_policy Defines the policy to fill the intermediate pixels.
     */
    void configure(const ICLTensor *input, ICLTensor *output, ICLTensor *slope);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPreluLayerKernel
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
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    ICLTensor       *_slope_gpu;
    bool       _run_in_place;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLPRELULAYERKERNEL_H__ */
