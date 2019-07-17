#include "arm_compute/graph/nodes/PreluLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
PreluLayerNode::PreluLayerNode()
{
    _input_edges.resize(2, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

TensorDescriptor PreluLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor)
{
    TensorDescriptor output_descriptor = input_descriptor;

    return output_descriptor;
}

bool PreluLayerNode::forward_descriptors()
{
    if((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor PreluLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    return src->desc();
}

NodeType PreluLayerNode::type() const
{
    return NodeType::PreluLayer;
}

void PreluLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
