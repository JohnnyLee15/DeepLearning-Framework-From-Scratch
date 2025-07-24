#include "core/layers/Flatten.h"
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"

void Flatten::forwardGpu(const Tensor &input, GpuCommandBuffer cmdBufVoid) {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    const vector<size_t> &shape = input.getShape();
    
    checkInputSize(shape);
    size_t batchSize = shape[0];

    inShape[0] = batchSize;
    outShape[0] = batchSize;
    input.copyGpu(output, cmdBuf);
    output.reShapeInPlace(outShape);
}