#include "arm_compute/core/CL/kernels/CLPreluLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "arm_compute/runtime/CL/CLTensor.h"
namespace arm_compute
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QASYMM8,
                                    "For QASYMM8 only logistic, relu, lower bounded relu and lower-upper bounded relu are supported");

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    if(output != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input);
    }

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    if(output != nullptr)
    {
        AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, input->valid_region());
    }
    else
    {
        window_changed = update_window_and_padding(win,
                                                   AccessWindowHorizontal(input, 0, num_elems_processed_per_iteration));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

CLPreluLayerKernel::CLPreluLayerKernel()
    : _input(nullptr), _output(nullptr), _slope_gpu(nullptr), _run_in_place(false)
{
}

Status CLPreluLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    const bool run_in_place = (output == nullptr) || (output == input);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (run_in_place) ? nullptr : output->clone().get()).first);
    return Status{};
}

void CLPreluLayerKernel::configure(const ICLTensor *input, ICLTensor *output, ICLTensor *slope)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _input                                     = input;
    _output                                    = output;
    _slope_gpu = slope;

    const DataLayout data_layout = input->info()->data_layout();

    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    
    TensorShape output_shape = arm_compute::TensorShape(input->info()->dimension(idx_width),
                                                        input->info()->dimension(idx_height), input->info()->dimension(idx_channel), 1U);

    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());
    output->info()->set_data_layout(data_layout);
    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    const DataType     dt                                = input->info()->data_type();

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CLPreluLayerKernel::validate(input->info(), output->info()));

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(dt)));
    build_opts.add_option(("-DSELECT_DATA_TYPE=" + get_cl_select_type_from_data_type(dt)));
    build_opts.add_option(("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("prelu_layer_" + lower_string(string_from_data_layout(input->info()->data_layout())), build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (_run_in_place) ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "prelu_layer_";
    _config_id += string_from_data_type(input->info()->data_type());
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(input->info()->data_layout()));
}

void CLPreluLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();
    Window vector_slice = window.first_slice_window_1D();
    vector_slice.set(Window::DimX, Window::Dimension(0, 0, 0));

    unsigned int include_output = (!_run_in_place) ? 1 : 0;
    unsigned int idx            = (1 + include_output) * num_arguments_per_3D_tensor();
    add_1D_tensor_argument(idx, _slope_gpu, vector_slice);
    do
    {
        idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        if(!_run_in_place)
        {
            add_3D_tensor_argument(idx, _output, slice);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}

} // namespace arm_compute
