#include "core/Layer.h"

void Layer::downloadOutputFromGpu() {}
void Layer::forwardGpu(const Tensor &prevActivations, id<MTLCommandBuffer> cmdBuf) {}

void Layer::backpropGpu(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    id<MTLCommandBuffer> cmdBuf
) {}