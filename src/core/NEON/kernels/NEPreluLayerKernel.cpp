#include "arm_compute/core/NEON/kernels/NEPreluLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
std::pair<Status, Window> validate_and_configure_window_nchw(ITensorInfo *input, ITensorInfo *output, int num_elems_processed_per_iteration_x)
{
    const int              num_elems_processed_per_iteration_x_out = num_elems_processed_per_iteration_x;
    Window                 win                                     = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x_out));
    AccessWindowRectangle  input_access(input, 0, 0, num_elems_processed_per_iteration_x, 1, 0.5f, 0.5f);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration_x_out);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, output->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window_nhwc(ITensorInfo *input, ITensorInfo *output, int num_elems_processed_per_iteration_x)
{
    Window                 win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration_x);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration_x);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, output->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, int num_elems_processed_per_iteration_x)
{
    std::pair<Status, Window> win_config;
    switch(input->data_layout())
    {
        case DataLayout::NCHW:
            win_config = validate_and_configure_window_nchw(input, output, num_elems_processed_per_iteration_x);
            break;
        case DataLayout::NHWC:
            win_config = validate_and_configure_window_nhwc(input, output, num_elems_processed_per_iteration_x);
            break;
        default:
            win_config = std::make_pair(ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported data layout!"), Window{});
    }

    return win_config;
}
} // namespace
NEPreluLayerKernel::NEPreluLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _num_elems_processed_per_iteration_x()
{
}

Status NEPreluLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    const DataLayout data_layout = input->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    //ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.x() != 2 || info.y() != 2, "Only stride 2 is supported");
    //ARM_COMPUTE_RETURN_ERROR_ON_MSG(policy != InterpolationPolicy::NEAREST_NEIGHBOR, "Only nearest neighbor policy supported");

    // Check output if configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_width) != input->dimension(idx_width));
        ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_height) != input->dimension(idx_height));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    const int num_elems_processed_per_iteration_x = 16 / input->element_size();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(),
                                                              output->clone().get(), num_elems_processed_per_iteration_x)
                                .first);
    return Status{};
}

void NEPreluLayerKernel::prelu_f32_nchw(const arm_compute::Window &window)
{
    Window window_in(window);
    window_in.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _num_elems_processed_per_iteration_x));
    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    Iterator  input(_input, window_in);
    Iterator  output(_output, window_out);

    execute_window_loop(window_out, [&](const Coordinates &id)
    {
        const float32x4_t data      = vld1q_f32(reinterpret_cast<const float *>(input.ptr()));
        float slope_value = *reinterpret_cast<const float*>(_slope->buffer() + _slope->info()->offset_element_in_bytes(Coordinates(id.z())));
        const float32x4_t data_out = { fmaxf(vgetq_lane_f32(data, 0), 0.0F) + slope_value * fminf(vgetq_lane_f32(data, 0), 0.0F),
                                       fmaxf(vgetq_lane_f32(data, 1), 0.0F) + slope_value * fminf(vgetq_lane_f32(data, 1), 0.0F),
                                       fmaxf(vgetq_lane_f32(data, 2), 0.0F) + slope_value * fminf(vgetq_lane_f32(data, 2), 0.0F),
                                       fmaxf(vgetq_lane_f32(data, 3), 0.0F) + slope_value * fminf(vgetq_lane_f32(data, 3), 0.0F) };
        auto              out       = reinterpret_cast<float *>(output.ptr());
        vst1q_f32(out, data_out);
    },
    input, output);
}

void NEPreluLayerKernel::prelu_f32_nhwc(const arm_compute::Window &window)
{
    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    window_out.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(2), 1));

    Iterator input(_input, window);
    Iterator output(_output, window_out);

    const int offset_y_out = _output->info()->strides_in_bytes().y() / sizeof(float);
    const int offset_z_out = _output->info()->strides_in_bytes().z() / sizeof(float);

    execute_window_loop(window_out, [&](const Coordinates &)
    {
        const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(input.ptr()));
        auto              out  = reinterpret_cast<float *>(output.ptr());

        vst1q_f32(out, data);
        vst1q_f32(out + offset_y_out, data);
        vst1q_f32(out + offset_z_out, data);
        vst1q_f32(out + offset_y_out + offset_z_out, data);
    },
    input, output);
}

void NEPreluLayerKernel::prelu_qasymm8_nchw(const arm_compute::Window &window)
{
    Window window_in(window);
    window_in.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _num_elems_processed_per_iteration_x));

    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));

    Iterator  input(_input, window_in);
    Iterator  output(_output, window_out);
    const int offset_y_out = _output->info()->strides_in_bytes().y() / sizeof(uint8_t);

    execute_window_loop(window_out, [&](const Coordinates &)
    {
        const uint8x16_t data      = vld1q_u8(reinterpret_cast<const uint8_t *>(input.ptr()));
        const uint8x16_t data_out1 = { vgetq_lane_u8(data, 0), vgetq_lane_u8(data, 0), vgetq_lane_u8(data, 1), vgetq_lane_u8(data, 1),
                                       vgetq_lane_u8(data, 2), vgetq_lane_u8(data, 2), vgetq_lane_u8(data, 3), vgetq_lane_u8(data, 3),
                                       vgetq_lane_u8(data, 4), vgetq_lane_u8(data, 4), vgetq_lane_u8(data, 5), vgetq_lane_u8(data, 5),
                                       vgetq_lane_u8(data, 6), vgetq_lane_u8(data, 6), vgetq_lane_u8(data, 7), vgetq_lane_u8(data, 7)
                                     };
        const uint8x16_t data_out2 =
        {
            vgetq_lane_u8(data, 8), vgetq_lane_u8(data, 8), vgetq_lane_u8(data, 9), vgetq_lane_u8(data, 9),
            vgetq_lane_u8(data, 10), vgetq_lane_u8(data, 10), vgetq_lane_u8(data, 11), vgetq_lane_u8(data, 11),
            vgetq_lane_u8(data, 12), vgetq_lane_u8(data, 12), vgetq_lane_u8(data, 13), vgetq_lane_u8(data, 13),
            vgetq_lane_u8(data, 14), vgetq_lane_u8(data, 14), vgetq_lane_u8(data, 15), vgetq_lane_u8(data, 15)
        };
        auto out = reinterpret_cast<uint8_t *>(output.ptr());

        vst1q_u8(out, data_out1);
        vst1q_u8(out + 16, data_out2);
        vst1q_u8(out + offset_y_out, data_out1);
        vst1q_u8(out + offset_y_out + 16, data_out2);
    },
    input, output);
}

void NEPreluLayerKernel::prelu_qasymm8_nhwc(const arm_compute::Window &window)
{
    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    window_out.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(2), 1));

    Iterator input(_input, window);
    Iterator output(_output, window_out);

    const int offset_y_out = _output->info()->strides_in_bytes().y() / sizeof(uint8_t);
    const int offset_z_out = _output->info()->strides_in_bytes().z() / sizeof(uint8_t);
    execute_window_loop(window_out, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(reinterpret_cast<const uint8_t *>(input.ptr()));
        auto             out  = reinterpret_cast<uint8_t *>(output.ptr());

        vst1q_u8(out, data);
        vst1q_u8(out + offset_y_out, data);
        vst1q_u8(out + offset_z_out, data);
        vst1q_u8(out + offset_y_out + offset_z_out, data);
    },
    input, output);
}

void NEPreluLayerKernel::prelu_f16_nchw(const arm_compute::Window &window)
{
    ARM_COMPUTE_UNUSED(window);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Window window_in(window);
    window_in.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _num_elems_processed_per_iteration_x));

    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));

    Iterator  input(_input, window_in);
    Iterator  output(_output, window_out);
    const int offset_y_out = _output->info()->strides_in_bytes().y() / sizeof(float16_t);

    execute_window_loop(window_out, [&](const Coordinates &)
    {
        const float16x8_t data      = vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr()));
        const float16x8_t data_out1 = { vgetq_lane_f16(data, 0), vgetq_lane_f16(data, 0), vgetq_lane_f16(data, 1), vgetq_lane_f16(data, 1),
                                        vgetq_lane_f16(data, 2), vgetq_lane_f16(data, 2), vgetq_lane_f16(data, 3), vgetq_lane_f16(data, 3)
                                      };
        const float16x8_t data_out2 = { vgetq_lane_f16(data, 4), vgetq_lane_f16(data, 4), vgetq_lane_f16(data, 5), vgetq_lane_f16(data, 5),
                                        vgetq_lane_f16(data, 6), vgetq_lane_f16(data, 6), vgetq_lane_f16(data, 7), vgetq_lane_f16(data, 7)
                                      };
        auto out = reinterpret_cast<float16_t *>(output.ptr());

        vst1q_f16(out, data_out1);
        vst1q_f16(out + 8, data_out2);
        vst1q_f16(out + offset_y_out, data_out1);
        vst1q_f16(out + offset_y_out + 8, data_out2);
    },
    input, output);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
}

void NEPreluLayerKernel::prelu_f16_nhwc(const arm_compute::Window &window)
{
    ARM_COMPUTE_UNUSED(window);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    window_out.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(2), 1));

    Iterator  input(_input, window);
    Iterator  output(_output, window_out);
    const int offset_y_out = _output->info()->strides_in_bytes().y() / sizeof(float16_t);
    const int offset_z_out = _output->info()->strides_in_bytes().z() / sizeof(float16_t);

    execute_window_loop(window_out, [&](const Coordinates &)
    {
        const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr()));
        auto              out  = reinterpret_cast<float16_t *>(output.ptr());

        vst1q_f16(out, data);
        vst1q_f16(out + offset_y_out, data);
        vst1q_f16(out + offset_z_out, data);
        vst1q_f16(out + offset_y_out + offset_z_out, data);
    },
    input, output);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
}

void NEPreluLayerKernel::configure(const ITensor *input, ITensor *output, ITensor *slope)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _input  = input;
    _output = output;
    _slope   = slope;

    const DataLayout data_layout = input->info()->data_layout();

    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    
    TensorShape output_shape = arm_compute::TensorShape(input->info()->dimension(idx_width),
                                                        input->info()->dimension(idx_height), input->info()->dimension(idx_channel), 1U);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());
    output->info()->set_data_layout(data_layout);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(NEPreluLayerKernel::validate(input->info(), output->info()));

    _num_elems_processed_per_iteration_x = 16 / output->info()->element_size();

    switch(data_layout)
    {
        case DataLayout::NCHW:
        {
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    _func = &NEPreluLayerKernel::prelu_qasymm8_nchw;
                    break;
                case DataType::F32:
                    _func = &NEPreluLayerKernel::prelu_f32_nchw;
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    _func = &NEPreluLayerKernel::prelu_f16_nchw;
                    break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        }
        case DataLayout::NHWC:
        {
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8:
                    _func = &NEPreluLayerKernel::prelu_qasymm8_nhwc;
                    break;
                case DataType::F32:
                    _func = &NEPreluLayerKernel::prelu_f32_nhwc;
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    _func = &NEPreluLayerKernel::prelu_f16_nhwc;
                    break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    // Configure window
    std::pair<Status, Window> win_config = validate_and_configure_window(input->info(),
                                                                         output->info(),
                                                                         _num_elems_processed_per_iteration_x);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

void NEPreluLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute
