#ifndef __ARM_COMPUTE_NEPRELULAYERKERNEL_H__
#define __ARM_COMPUTE_NEPRELULAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the Prelu layer kernel.*/
class NEPreluLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEPreluLayerKernel";
    }
    /** Default constructor */
    NEPreluLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPreluLayerKernel(const NEPreluLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPreluLayerKernel &operator=(const NEPreluLayerKernel &) = delete;
    /** Default Move Constructor. */
    NEPreluLayerKernel(NEPreluLayerKernel &&) = default;
    /** Default move assignment operator */
    NEPreluLayerKernel &operator=(NEPreluLayerKernel &&) = default;
    /** Default destructor */
    ~NEPreluLayerKernel() = default;
    /** Set the input output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/F16/F32.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     * @param[in]  info   Contains stride information described in @ref Size2D.
     * @param[in]  policy Defines the policy to fill the intermediate pixels.
     *
     */
    void configure(const ITensor *input, ITensor *output, ITensor *slope);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPreluLayerKernel
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/F16/F32.
     * @param[in] output Destination tensor info. Data types supported: same as @p input.
     * @param[in] info   Contains stride information described in @ref Size2D.
     * @param[in] policy Defines the policy to fill the intermediate pixels.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Function to run prelu layer for FP32 (NCHW)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    void prelu_f32_nchw(const Window &window);
    /** Function to run prelu layer for FP32 (NHWC)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    void prelu_f32_nhwc(const Window &window);
    /** Function to run prelu layer for FP16 (NCHW)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    void prelu_f16_nchw(const Window &window);
    /** Function to run prelu layer for FP16 (NHWC)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    void prelu_f16_nhwc(const Window &window);
    /** Function to run prelu layer for QASYMM8 (NCHW)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    void prelu_qasymm8_nchw(const Window &window);
    /** Function to run prelu layer for QASYMM8 (NHWC)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    void prelu_qasymm8_nhwc(const Window &window);
    /** Common signature for all the prelu layer functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using PreluFunctionPtr = void (NEPreluLayerKernel::*)(const Window &window);

private:
    PreluFunctionPtr _func;
    const ITensor      *_input;
    ITensor            *_output;
    unsigned int        _num_elems_processed_per_iteration_x;
    ITensor *_slope;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEPRELULAYERKERNEL_H__ */
