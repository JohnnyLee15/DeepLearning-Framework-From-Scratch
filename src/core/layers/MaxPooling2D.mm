#include "core/layers/MaxPooling2D.h"
#include "core/tensor/Tensor.h"
#include "core/activations/Activation.h"
#include "core/gpu/GpuEngine.h"
#include <climits>
#include <cfloat> 

void MaxPooling2D::forwardGpu(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    if (input.getShape()[0] != paddedInput.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    GpuEngine::fillInt(maxIndicesGpu.getBuffer(), (uint32_t) pooledOutput.getSize(), cmdBuf, UINT_MAX);
    const Tensor &inputFwd = input.padIfNeededGpu(paddedInput, winIn, padding, cmdBuf, -FLT_MAX);
    inputFwd.maxPool2dGpu(maxIndicesGpu, kRows, kCols, stride, pooledOutput, cmdBuf, winIn);
}

void MaxPooling2D::backpropGpu(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &outputGradients,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    // Add error checking
    (void)learningRate;
    (void)isFirstLayer;
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    outputGradients.maxPool2dGradGpu(maxIndicesGpu, dX, cmdBuf);
}
