#include "core/data/Dataset.h"
#include "core/data/Batch.h"

Dataset::Dataset(const Tensor &features, const vector<float> &targets) :
    features(&features), targets(&targets), binData(nullptr), isMemory(true) {}

Dataset::Dataset(const Tensor &features) : 
    features(&features), targets(nullptr), binData(nullptr), isMemory(true) {}

Dataset::Dataset(const BinLoader &binData) : 
    features(nullptr), targets(nullptr), binData(&binData), isMemory(false) {}


size_t Dataset::sampleCount() const {
    if (isMemory) 
        return features->getShape()[0];

    return binData->sampleCount();
}

size_t Dataset::xSize() const {
    if (isMemory)
        return features->getSize();

    return binData->xSize();
}

const vector<size_t>& Dataset::xShape() const {
    if (isMemory)
        return features->getShape();

    return binData->getShapeX();
}

void Dataset::fillBatch(
    Batch &batch,
    size_t start, 
    size_t end, 
    const vector<size_t> &shuffledIndices
) const {
    size_t batchSize = end - start;
    batch.setBatchSize(batchSize);
    batch.setBatchIndices(start, end, shuffledIndices);

    if (isMemory) {
        fillBatchFromMemory(batch);
    } else {
        binData->fillBatch(batch);
    }
    batch.uploadToGpu();
}

void Dataset::fillBatchFromMemory(Batch &batch) const {
    size_t batchSize = batch.getBatchSize();
    size_t elementSize = batch.getData().getSize() / batchSize;

    vector<float> &xBatchFlat = batch.getData().getFlat();
    vector<float> &yBatchFlat = batch.getTargets().getFlat();
    const vector<float> &xTrainFlat = features->getFlat();
    const vector<float> &yTrainFlat = *targets;
    const vector<size_t> &indices = batch.getIndices();
    
    #pragma omp parallel for
    for (size_t i = 0; i < batchSize; i++) {
        size_t rdIdx = indices[i];
        memcpy(
            xBatchFlat.data() + (i * elementSize), 
            xTrainFlat.data() + (rdIdx * elementSize), 
            elementSize * sizeof(float)
        );
        yBatchFlat[i] = yTrainFlat[rdIdx];
    }
}

void Dataset::fillInferenceBatch(
    Tensor &batch,
    size_t start,
    size_t batchSize
) const {
    vector<size_t> batchShape = batch.getShape();
    batchShape[0] = batchSize;
    batch.reShapeInPlace(batchShape);

    if (isMemory) {
       fillInfBatchFromMemory(batch, start, batchSize);
    } else {
        binData->fillInfBatch(batch, start, batchSize);
    }
}

void Dataset::fillInfBatchFromMemory(
    Tensor &batch,
    size_t start,
    size_t batchSize
) const {
    size_t sampleFloats = batch.getSize() / batch.getShape()[0];
    size_t sampleStartFloat = start * sampleFloats;
    size_t bytes = batchSize * sampleFloats * sizeof(float);
    memcpy(batch.getFlat().data(), features->getFlat().data() + sampleStartFloat, bytes);
}
