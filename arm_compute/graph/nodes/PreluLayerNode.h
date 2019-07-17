#ifndef __ARM_COMPUTE_GRAPH_PRELU_LAYER_NODE_H__
#define __ARM_COMPUTE_GRAPH_PRELU_LAYER_NODE_H__

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Prelu Layer node */
class PreluLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] info              Stride info
     * @param[in] upsampling_policy Upsampling policy
     */
    PreluLayerNode();
    /** Stride info metadata accessor
     *
     * @return The stride info of the layer
     */
    /** Upsampling policy metadata accessor
     *
     * @return The upsampling policy of the layer
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    //ITensorAccessorUPtr _slope;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_PRELU_LAYER_NODE_H__ */
