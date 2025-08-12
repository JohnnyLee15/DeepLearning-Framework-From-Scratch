#include "core/layers/Dropout.h"
#include <random>
#include <algorithm>
#include <omp.h>

Dropout::Dropout() : rate(0.0f) {}

Dropout::Dropout(float r) {
    rate = max(0.0f, min(r, 0.999999f));
}

void Dropout::build(const vector<size_t> &inShape, bool isInference) {
    Layer::build(inShape);
    output = Tensor(inShape);

    if (isInference) {
        dX = Tensor();
    } else {
        dX = Tensor(inShape);
        mask = Tensor(inShape);
    }
}

vector<size_t> Dropout::getBuildOutShape(const vector<size_t> &inShape) const {
    vector<size_t> outShape = inShape;
    outShape[0] = getMaxBatchSize();
    return outShape;
}

vector<uint32_t> Dropout::generateThreadSeeds() const {
    size_t numSeeds = omp_get_max_threads();
    vector<uint32_t> seeds(numSeeds);
    random_device rd;
    for (size_t i = 0; i < numSeeds; i++) {
        seeds[i] = rd();
    }

    return seeds;
}

void Dropout::generateMask() {
    vector<uint32_t> seeds = generateThreadSeeds();
    size_t size = dX.getSize();

    float pKeep = 1.0f - rate;
    float scale = 1.0f/pKeep;

    vector<float> &maskFlat = mask.getFlat();

    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        mt19937 generator(seeds[thread]);
        bernoulli_distribution distribution(pKeep);

        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            maskFlat[i] = distribution(generator) * scale;
        }
    }
}

void Dropout::reShapeBatch(size_t currBatchSize) {
    vector<size_t> newSize = output.getShape();
    newSize[0] = currBatchSize;

    output.reShapeInPlace(newSize);
    if (dX.getSize() > 0) {
        dX.reShapeInPlace(newSize);
        mask.reShapeInPlace(newSize);
    }
}

void Dropout::forward(const Tensor &input) {
    if (input.getShape()[0] != output.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    if (dX.getSize() == 0) {
        output = input;
        return;
    }

    generateMask();
    input.applyMask(mask, output);
}

void Dropout::backprop(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer
) {
    (void)prevActivations;
    (void)learningRate;
    (void)isFirstLayer;
    
    grad.applyMask(mask, dX);
}

void Dropout::writeBinInternal(ofstream &modelBin) const {
    modelBin.write((char*) &rate, sizeof(float));
}

void Dropout::loadFromBin(ifstream &modelBin) {
    modelBin.read((char*) &rate, sizeof(float));
}

const Tensor& Dropout::getOutput() const {
    return output;
}

Tensor& Dropout::getOutputGradient() {
    return dX;
}

Layer::Encodings Dropout::getEncoding() const {
    return Layer::Encodings::Dropout;
}

Layer* Dropout::clone() const {
    return new Dropout(*this);
}