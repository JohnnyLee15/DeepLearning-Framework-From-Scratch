#include "core/activations/Activation.h"
#include "core/tensor/Tensor.h"
#include "utils/ConsoleUtils.h"

void Activation::backpropGpu(
    const Tensor &activations,  
    Tensor &grad,        
    GpuCommandBuffer cmdBufVoid
) const {
    ConsoleUtils::fatalError(
        "Fused back-prop is not supported for this activation."
    );
}