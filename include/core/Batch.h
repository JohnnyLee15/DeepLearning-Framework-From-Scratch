#pragma once

#include <vector>
#include "core/Tensor.h"

class Loss;
class DenseLayer;

using namespace std;

class Batch {
    private:
        // Instance Variables
        size_t batchSize;
        vector<size_t> indices;
        vector<double> targets;
        Tensor data;

    public:
        // Constructor
        Batch(size_t, size_t);

        // Methods
        void setBatch(const Tensor&, const vector<double> &);
        void setBatchIndices(size_t, size_t, const vector<size_t>&);
        const Tensor& getData() const;
        const vector<double>& getTargets() const;
        size_t getSize() const;
        const vector<size_t>& getIndices() const;
};