#include "core/layers/Dense.h"
#include "core/tensor/Tensor.h"
#include "core/tensor/Matrix.h"
#include "core/tensor/MatrixT.h"
#include "core/activations/Activation.h"
#include "core/gpu/GpuEngine.h"

void Dense::forwardGpu(const Tensor &prevActivations, GpuCommandBuffer cmdBufVoid) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    if (prevActivations.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(prevActivations.getShape()[0]);
    }

    prevActivations.M().mmTGpu(weights.M().T(), preActivations, cmdBuf);
    preActivations.M().addToRowsGpu(biases, cmdBuf);
    activation->activateGpu(preActivations, activations, (GpuCommandBuffer) cmdBuf); 
}

void Dense::backpropGpu(
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

    Matrix gradMat = grad.M();
    size_t batchSize = gradMat.getNumRows();
    float scaleFactor = -learningRate/batchSize;

    gradMat.T().mTmGpu(prevActivations.M(), dW, cmdBuf);
    gradMat.colSumsGpu(dB, cmdBuf);

    weights.applyGradGpu(dW, scaleFactor, cmdBuf);
    biases.applyGradGpu(dB, scaleFactor, cmdBuf);

    if (!isFirstLayer) {
        gradMat.mmGpu(weights, dX, cmdBuf);
    }
}

void Dense::writeToBinGpu() {
    weights.downloadFromGpu();
    biases.downloadFromGpu();
}