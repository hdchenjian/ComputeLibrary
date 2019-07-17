#include "helpers.h"

#define MUL_OP(a, b) ((a) * (b))

#if defined(VEC_SIZE) && defined(DATA_TYPE)

#define TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define SELECT_TYPE VEC_DATA_TYPE(SELECT_DATA_TYPE, VEC_SIZE)

__kernel void prelu_layer_nchw(TENSOR3D_DECLARATION(input), TENSOR3D_DECLARATION(output), VECTOR_DECLARATION(slope)) {
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
    Vector slope = CONVERT_TO_VECTOR_STRUCT(slope);

    // Load data
    TYPE data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input.ptr);
    const int current_slice = get_global_id(2);

    TYPE _slope = *((__global DATA_TYPE *)(slope.ptr + current_slice * slope.stride_x));
    //fmaxf(vgetq_lane_f32(data, 0), 0.0F) + slope_value * fminf(vgetq_lane_f32(data, 0), 0.0F);

    data = select(MUL_OP((TYPE)_slope, data), data, CONVERT(data > (TYPE)0, SELECT_TYPE));

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)output.ptr);
}

#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) */
