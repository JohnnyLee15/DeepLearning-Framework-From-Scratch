#pragma once

#include <vector>
#include "utils/Matrix.h"
#include <cstdint>

class Activation; 

using namespace std;

class Layer {
    private:

        // Constants
        static const double HE_INT_GAIN;

        // Instance Variables
        Matrix activations;
        Matrix preActivations;
        Activation *activation;
        Matrix weights;
        vector<double> biases;

        // Methods
        void initWeights(size_t, size_t);
        vector<uint32_t> generateThreadSeeds() const;

    public:
        Layer(int, int, Activation*);
        void calActivations(const vector<double>&);
        void calActivations(const Matrix&);
        const Matrix getActivations() const;
        const Matrix getPreActivations() const;
        Activation* getActivation() const;
        void updateLayerParameters(const Matrix&, double, const Matrix&);
        Matrix updateOutputGradient(const Matrix&, const Matrix&, Activation*);
};