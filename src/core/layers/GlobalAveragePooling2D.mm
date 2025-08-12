#include "core/layers/GlobalAveragePooling2D.h"

void GlobalAveragePooling2D::forwardGpu(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    if (input.getShape()[0] != output.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    input.globalAvgPool2dGpu(output, cmdBuf);
}

void GlobalAveragePooling2D::backpropGpu(
    const Tensor &input,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    (void)input;
    (void)learningRate;
    (void)isFirstLayer;

    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    grad.globalAvgPool2dGradGpu(dX, cmdBuf);
}