#include "core/data/Batch.h"
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "core/activations/Activation.h"
#include "core/tensor/Matrix.h"
#include "core/gpu/GpuEngine.h"
#include <cstring>

Batch::Batch(size_t numLayers, size_t batchSize) :
    batchSize(batchSize),
    indices(batchSize),
    targets({batchSize})
{}

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

void Batch::ensureGpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            data.uploadToGpu();
            targets.uploadToGpu();
        #endif
    }
}

void Batch::setBatch(
    const Tensor &train,
    const vector<float> &trainLabels
) {
    const vector<size_t> &trainShape = train.getShape();
    vector<size_t> batchShape = trainShape;
    batchShape[0] = batchSize;
    data = Tensor(batchShape);
    
    size_t elementSize = data.getSize() / batchSize;

    vector<float> &batchFlat = data.getFlat();
    vector<float> &targetsFlat = targets.getFlat();
    const vector<float> &trainFlat = train.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < batchSize; i++) {
        size_t rdIdx = indices[i];
        memcpy(
            batchFlat.data() + (i * elementSize), 
            trainFlat.data() + (rdIdx * elementSize), 
            elementSize * sizeof(float)
        );
        targetsFlat[i] = trainLabels[rdIdx];
    }
    ensureGpu();
}

const Tensor& Batch::getData() const {
    return data;
}

const Tensor& Batch::getTargets() const {
    return targets;
}

size_t Batch::getSize() const {
    return data.getShape()[0];
}

const vector<size_t>& Batch::getIndices() const {
    return indices;
}