#include "core/layers/Dense.h"
#include "core/tensor/Tensor.h"
#include "core/tensor/Matrix.h"
#include "core/tensor/MatrixT.h"
#include "core/activations/Activation.h"
#include "core/gpu/GpuEngine.h"
#include <iostream>

void Dense::forwardGpu(const Tensor &prevActivations, GpuCommandBuffer cmdBufVoid) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    if (prevActivations.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(prevActivations.getShape()[0]);
    }

    if (executionMode == GPU_FAST) {
        prevActivations.M().mmTBiasReLU(weights.M().T(), activations, biases, cmdBuf);
    } else {
        prevActivations.M().mmTGpu(weights.M().T(), preActivations, cmdBuf);
        preActivations.M().addToRowsGpu(biases, cmdBuf);
        activation->activateGpu(preActivations, activations, (GpuCommandBuffer) cmdBuf); 
    }
}

void Dense::backpropGpu(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    if (executionMode == GPU_FAST) {
        backpropGpuFast(prevActivations, learningRate, grad, isFirstLayer, cmdBufVoid);
    } else {
        backpropGpuNaive(prevActivations, learningRate, grad, isFirstLayer, cmdBufVoid);
    }
}

void Dense::backpropGpuFast(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    activation->backpropGpu(activations, grad, cmdBuf);

    if (!isFirstLayer) {
        grad.M().mmGpu(weights, dX, (id<MTLCommandBuffer>) cmdBufVoid);
    }
    
    Matrix gradMat = grad.M();
    size_t batchSize = gradMat.getNumRows();
    float scaleFactor = -learningRate/batchSize;
    
    gradMat.applyBiasGradDense(biases, scaleFactor, cmdBuf);
    gradMat.T().applyWeightsGrad(prevActivations.M(), weights, scaleFactor, cmdBuf);
}

void Dense::backpropGpuNaive(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    if (!activation->isFused()) {
        activation->calculateGradientGpu(preActivations, dA, (GpuCommandBuffer) cmdBuf);
        grad.hadamardGpu(dA, cmdBuf);
    }

    if (!isFirstLayer) {
        grad.M().mmGpu(weights, dX, (id<MTLCommandBuffer>) cmdBufVoid);
    }

    Matrix gradMat = grad.M();
    size_t batchSize = gradMat.getNumRows();
    float scaleFactor = -learningRate/batchSize;

    gradMat.applyBiasGradDense(biases, scaleFactor, cmdBuf);
    gradMat.T().applyWeightsGrad(prevActivations.M(), weights, scaleFactor, cmdBuf);
}

void Dense::downloadOutputFromGpu() {
    activations.downloadFromGpu();
}