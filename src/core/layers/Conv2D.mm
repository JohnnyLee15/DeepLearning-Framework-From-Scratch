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

void Conv2D::backpropGpu(
    const Tensor &input,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    (void) isFirstLayer;
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    float scaleFactor = -learningRate / input.getShape()[0];

    activation->calculateGradientGpu(preActivations, dA, cmdBuf);
    grad.hadamardGpu(dA, cmdBuf);

    const Tensor &inputBwd = input.padIfNeededGpu(paddedInput, winIn, padding, cmdBuf);
    inputBwd.conv2dWeightsGpu(grad, numKernals, kRows, kCols, stride, dW, cmdBuf);
    grad.reduceSumBiasGpu(dB, cmdBuf);

    kernals.applyGradGpu(dW, scaleFactor, cmdBuf);
    biases.applyGradGpu(dB, scaleFactor, cmdBuf);

    grad.padAndUpsampleGradGpu(gradBuf, winGrad, stride, cmdBuf);
    gradBuf.conv2dInputGpu(kernals, dX, cmdBuf);
}