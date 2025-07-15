#pragma once

#include <vector>
#include "core/Tensor.h"

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
        Batch(size_t, size_t);

        // Methods
        void setBatch(const Tensor&, const vector<float> &);
        void setBatchIndices(size_t, size_t, const vector<size_t>&);
        Tensor& getData();
        const Tensor& getTargets() const;
        size_t getSize() const;
        const vector<size_t>& getIndices() const;
};