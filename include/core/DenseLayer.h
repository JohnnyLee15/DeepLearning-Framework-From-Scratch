#pragma once

#include <vector>
#include <cstdint>
#include "core/Matrix.h"
#include "core/Layer.h"

class Activation;

using namespace std;

class DenseLayer : public Layer {
    private:

        // Constants
        static const double HE_INT_GAIN;

        // Instance Variables
        Matrix activations;
        Matrix preActivations;
        Matrix weights;
        Matrix dZ;
        Activation *activation;
        vector<double> biases;


        // Methods
        void initWeights(size_t, size_t);
        vector<uint32_t> generateThreadSeeds() const;

    public:
        // Constructor
        DenseLayer(size_t, size_t, Activation*);

        // Methods
        void calActivations(const Matrix&) override;
        const Matrix getActivations() const override;
        Matrix getOutputGradient() const override;
        void backprop(const Matrix&, double, const Matrix&, bool) override;
        ~DenseLayer();
};