#include "core/layers/Layer.h"

void Layer::forwardGpu(const Tensor &prevActivations, GpuCommandBuffer cmdBuf) {}

void Layer::backpropGpu(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBuf
) {}