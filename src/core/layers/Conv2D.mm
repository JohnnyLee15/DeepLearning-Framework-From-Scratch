#include "core/layers/Conv2D.h"
#include "core/tensor/Tensor.h"
#include "core/activations/Activation.h"
#include "core/gpu/GpuEngine.h"

void Conv2D::forwardGpu(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    if (input.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    const Tensor &inputFwd = input.padIfNeededGpu(paddedInput, winIn, padding, cmdBuf);
    inputFwd.conv2dForwardGpu(kernals, stride, preActivations, biases, cmdBuf);
    activation->activateGpu(preActivations, activations, (GpuCommandBuffer) cmdBuf);
}