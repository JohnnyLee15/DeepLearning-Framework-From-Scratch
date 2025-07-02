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
        vector<double> rescaledTargets;
        Tensor data;
        Tensor rescaledOutput;

    public:
        // Constructor
        Batch(size_t, size_t);

        // Methods
        void setBatch(const Tensor&, const vector<double> &);
        void setBatchIndices(size_t, size_t, const vector<size_t>&);
        const Tensor& getData() const;
        const vector<double>& getTargets() const;
        void writeBatchPredictions(vector<double>&, const Tensor&) const;
        size_t getCorrectPredictions(const vector<double>&) const; 
        void setRescaledOutput(const Tensor&);
        void setRescaledTargets(const vector<double>&);
        const Tensor& getRescaledOutput() const;
        const vector<double>& getRescaledTargets() const;
};