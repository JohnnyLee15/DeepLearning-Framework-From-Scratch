#include "core/layers/Conv2D.h"
#include "core/tensor/Tensor.h"
#include "core/activations/Activation.h"
#include "core/gpu/GpuEngine.h"
#include "utils/Im2ColUtils.h"
#include "core/tensor/Matrix.h"
#include "core/tensor/MatrixT.h"
#include <iostream>

void Conv2D::forwardGpu(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    if (input.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }
    
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    const Tensor &inputFwd = input.padIfNeededGpu(paddedInput, winIn, padding, cmdBuf);
    
    if (executionMode == GPU_FAST) {
        forwardGpuFast(inputFwd, cmdBufVoid);
    } else {
        forwardGpuNaive(inputFwd, cmdBufVoid);
    }
}

void Conv2D::forwardGpuFast(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    Im2ColUtils::im2Col(input, im2ColInBuf, kRows, kCols, stride, winIn, cmdBuf);
    preActivations.reShapeInPlace(im2ColPreActShape);
    im2ColInBuf.M().mmBiasReLU(im2ColKBuf, activations, biases, cmdBuf);
    preActivations.reShapeInPlace(preActTensorShape);
}

void Conv2D::forwardGpuNaive(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    input.conv2dForwardGpu(kernals, stride, preActivations, biases, cmdBuf);
    activation->activateGpu(preActivations, activations, cmdBufVoid);
}

void Conv2D::backpropGpu(
    const Tensor &input,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    if (executionMode == GPU_FAST) {
        backpropGpuFast(input, learningRate, grad, isFirstLayer, cmdBufVoid);
    } else {
        backpropGpuNaive(input, learningRate, grad, isFirstLayer, cmdBufVoid);
    }
}

void Conv2D::backpropGpuFast(
    const Tensor &input,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    float scaleFactor = -learningRate / input.getShape()[0];

    activation->backpropGpu(activations, grad, cmdBuf);
    grad.applyBiasGrad(biases, scaleFactor, cmdBuf);

    grad.reShapeInPlace(im2ColPreActShape);
    im2ColInBuf.M().T().applyKernelGrads(grad, im2ColKBuf, scaleFactor, cmdBuf);

    if (!isFirstLayer) {
        grad.M().mmTGpu(im2ColKBuf.M().T(), gradIm2ColBuf, cmdBuf);
        Im2ColUtils::col2Im(
            gradIm2ColBuf, dX, preActTensorShape[1], preActTensorShape[2],
            kRows, kCols, stride, winIn.padTop, winIn.padLeft, cmdBuf
        );
    }

    grad.reShapeInPlace(preActTensorShape);
}

void Conv2D::backpropGpuNaive(
    const Tensor &input,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer,
    GpuCommandBuffer cmdBufVoid
) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)cmdBufVoid;
    float scaleFactor = -learningRate / input.getShape()[0];

    activation->calculateGradientGpu(preActivations, dA, cmdBufVoid);
    grad.hadamardGpu(dA, cmdBuf);

    const Tensor &inputBwd = input.padIfNeededGpu(paddedInput, winIn, padding, cmdBuf);
    inputBwd.conv2dWeightsGpu(grad, numKernals, kRows, kCols, stride, dW, cmdBuf);
    
    kernals.applyGradGpu(dW, scaleFactor, cmdBuf);
    grad.applyBiasGrad(biases, scaleFactor, cmdBuf);

    if (!isFirstLayer) {
        grad.padAndUpsampleGradGpu(gradBuf, winGrad, stride, cmdBuf);
        gradBuf.conv2dInputGpu(kernals, dX, cmdBuf);
    }
}