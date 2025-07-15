#include "activations/Activation.h"
#include "core/Tensor.h"

bool Activation::isFused() const {
    return false;
}