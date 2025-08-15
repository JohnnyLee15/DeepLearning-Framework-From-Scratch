#include "core/layers/Dropout.h"
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"

void Dropout::forwardGpu(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    if (input.getShape()[0] != output.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    if (dX.getSize() == 0) {
        output = input;
        return;
    }

    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    generateMask();
    mask.uploadToGpu();
    input.applyMaskGpu(mask, output, cmdBuf);
}

void Dropout::backpropGpu(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    (void)prevActivations;
    (void)learningRate;
    (void)isFirstLayer;

    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    grad.applyMaskGpu(mask, dX, cmdBuf);
}
