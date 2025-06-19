#pragma once

#include <vector>
#include "core/Matrix.h"

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
        Matrix data;
        Matrix rescaledOutput;

    public:
        // Constructor
        Batch(size_t, size_t);

        // Methods
        void setBatch(const Matrix&, const vector<double> &);
        void setBatchIndices(size_t, size_t, const vector<size_t>&);
        const Matrix& getData() const;
        const vector<double>& getTargets() const;
        void writeBatchPredictions(vector<double>&, const Matrix&) const;
        size_t getCorrectPredictions(const vector<double>&) const; 
        void setRescaledOutput(const Matrix&);
        void setRescaledTargets(const vector<double>&);
        const Matrix& getRescaledOutput() const;
        const vector<double>& getRescaledTargets() const;
};