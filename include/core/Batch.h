#pragma once

#include <vector>
#include "core/Matrix.h"

class Loss;
class Layer;

using namespace std;

class Batch {
    private:
        // Instance Variables
        size_t batchSize;
        vector<Matrix> layerActivations;
        vector<Matrix> layerPreActivations;
        vector<size_t> indices;
        vector<double> targets;
        vector<double> rescaledTargets;
        Matrix data;
        Matrix rescaledOutput;
        Matrix outputGradients;
        size_t writeActivationIdx;
        size_t writePreActivationIdx;

    public:
        // Constructor
        Batch(size_t, size_t);

        // Methods
        void setBatch(const Matrix&, const vector<double> &);
        void setBatchIndices(size_t, size_t, const vector<size_t>&);
        const Matrix& getData() const;
        void addLayerActivations(const Matrix&);
        void addLayerPreActivations(const Matrix&);
        const Matrix& getLayerActivation(size_t) const;
        const Matrix& getLayerPreActivation(size_t) const;
        const Matrix& getOutputGradients() const;
        const vector<double>& getTargets() const;
        void updateOutputGradients(const Matrix&);
        void calculateOutputGradients(const Layer&, const Loss*);
        void writeBatchPredictions(vector<double>&, const Matrix&) const;
        size_t getCorrectPredictions(const vector<double>&) const; 
        void setRescaledOutput(const Matrix&);
        void setRescaledTargets(const vector<double>&);
        const Matrix& getRescaledOutput() const;
        const vector<double>& getRescaledTargets() const;
};