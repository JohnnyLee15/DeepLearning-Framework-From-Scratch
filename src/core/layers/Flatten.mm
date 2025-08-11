#include "core/layers/Flatten.h"
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"

void Flatten::forwardGpu(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    const vector<size_t> &shape = input.getShape();
    
    checkInputSize(shape);
    size_t batchSize = shape[0];

    inShape[0] = batchSize;
    outShape[0] = batchSize;
    output = input;
    output.reShapeInPlace(outShape);
}

void Flatten::backpropGpu(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    // Add error checking
    (void)prevActivations;
    (void)learningRate;
    (void)isFirstLayer;
    dX = grad;
    dX.reShapeInPlace(inShape);
}