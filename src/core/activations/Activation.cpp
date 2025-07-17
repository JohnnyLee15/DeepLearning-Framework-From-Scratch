#include "core/activations/Activation.h"
#include "core/tensor/Tensor.h"

bool Activation::isFused() const {
    return false;
}