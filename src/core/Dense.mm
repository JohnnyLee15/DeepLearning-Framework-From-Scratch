#include "core/Dense.h"
#include "core/Tensor.h"
#include "core/Matrix.h"
#include "core/MatrixT.h"
#include "activations/Activation.h"

void Dense::forwardGpu(const Tensor &prevActivations, id<MTLCommandBuffer> cmdBuf) {
    if (prevActivations.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(prevActivations.getShape()[0]);
    }

    MatrixT weightsT = weights.M().T();
    Matrix prevActMat = prevActivations.M();

    prevActMat.mmTGpu(weightsT, preActivations, cmdBuf);
    prevActMat.addToRowsGpu(biases, cmdBuf);
    activation->activateGpu(preActivations, activations, cmdBuf); 
}

void Dense::backpropGpu(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    id<MTLCommandBuffer> cmdBuf
) {
    if (!activation->isFused()) {
        activation->calculateGradientGpu(preActivations, dA, cmdBuf);
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