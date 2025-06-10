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
        vector<int> indices;
        vector<double> targets;
        vector<double> rescaledTargets;
        Matrix data;
        Matrix rescaledOutput;
        Matrix outputGradients;
        int writeActivationIdx;
        int writePreActivationIdx;

    public:
        // Constructor
        Batch(int, int);

        // Methods
        void setBatch(const Matrix&, const vector<double> &);
        void setBatchIndices(int, int, const vector<int>&);
        const Matrix& getData() const;
        void addLayerActivations(const Matrix&);
        void addLayerPreActivations(const Matrix&);
        const Matrix& getLayerActivation(int) const;
        const Matrix& getLayerPreActivation(int) const;
        const Matrix& getOutputGradients() const;
        const vector<double>& getTargets() const;
        void updateOutputGradients(const Matrix&);
        void calculateOutputGradients(const Layer&, const Loss*);
        void writeBatchPredictions(vector<double>&, const Matrix&) const;
        int getCorrectPredictions(const vector<double>&) const; 
        void setRescaledOutput(const Matrix&);
        void setRescaledTargets(const vector<double>&);
        const Matrix& getRescaledOutput() const;
        const vector<double>& getRescaledTargets() const;
};