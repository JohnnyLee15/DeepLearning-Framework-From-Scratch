#pragma once

#include "core/tensor/Tensor.h"
#include "core/data/BinLoader.h"
#include <vector>

class Batch;


using namespace std;

class Dataset {
    private:
        // Instance Variables
        const Tensor *features;
        const vector<float> *targets;
        const BinLoader *binData;
        bool isMemory;

        // Methods
        void fillBatchFromMemory(Batch&) const;
        void fillInfBatchFromMemory(Tensor&, size_t, size_t) const;

    public:
        // Constructors
        Dataset(const Tensor&, const vector<float>&);
        Dataset(const Tensor&);
        Dataset(const BinLoader&);

        // Methods
        size_t sampleCount() const;
        size_t xSize() const;
        const vector<size_t> &xShape() const;
        void fillBatch(Batch&, size_t, size_t, const vector<size_t>&) const;
        void fillInferenceBatch(Tensor&, size_t, size_t) const;
        
};