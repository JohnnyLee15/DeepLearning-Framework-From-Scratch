#include "core/data/Batch.h"
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "core/activations/Activation.h"
#include "core/tensor/Matrix.h"
#include "core/gpu/GpuEngine.h"
#include <cstring>

Batch::Batch(size_t maxBatchSize, vector<size_t> xShape) :
    batchSize(maxBatchSize),
    indices(maxBatchSize),
    targets({maxBatchSize})
{
    xShape[0] = maxBatchSize;
    data = Tensor(xShape);
}

void Batch::setBatchSize(size_t currBatchSize) {
    batchSize = currBatchSize;
    vector<size_t> shape = data.getShape();
    shape[0] = currBatchSize;
    data.reShapeInPlace(shape);
}

void Batch::setBatchIndices(
    size_t start,
    size_t end,
    const vector<size_t> &shuffledIndices
) {
    #pragma omp parallel for
    for (size_t i = start; i < end; i++) {
        indices[i - start] = shuffledIndices[i];
    }
}

void Batch::uploadToGpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            data.uploadToGpu();
            targets.uploadToGpu();
        #endif
    }
}

const Tensor& Batch::getData() const {
    return data;
}

Tensor& Batch::getData() {
    return data;
}

const Tensor& Batch::getTargets() const {
    return targets;
}

Tensor& Batch::getTargets() {
    return targets;
}

size_t Batch::getBatchSize() const {
    return data.getShape()[0];
}

const vector<size_t>& Batch::getIndices() const {
    return indices;
}