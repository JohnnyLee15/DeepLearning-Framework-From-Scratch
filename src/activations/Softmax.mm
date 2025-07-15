#include "activations/Softmax.h"
#include "losses/SoftmaxCrossEntropy.h"

void Softmax::calculateGradientGpu(const Tensor &z, Tensor &dZ) const {
    SoftmaxCrossEntropy::checkInvalidGradientCall();
}
