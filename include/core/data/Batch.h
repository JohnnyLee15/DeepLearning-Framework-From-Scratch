#pragma once

#include <vector>
#include "core/tensor/Tensor.h"

class Loss;

using namespace std;

class Batch {
    private:
        // Instance Variables
        size_t batchSize;
        vector<size_t> indices;
        Tensor targets;
        Tensor data;

    public:
        // Constructor
        Batch(size_t, vector<size_t>);

        // Methods
        void setBatchSize(size_t);
        void setBatchIndices(size_t, size_t, const vector<size_t>&);
        void uploadToGpu();

        const Tensor& getData() const;
        Tensor& getData();

        const Tensor& getTargets() const;
        Tensor& getTargets();
        
        size_t getBatchSize() const;
        const vector<size_t>& getIndices() const;
};