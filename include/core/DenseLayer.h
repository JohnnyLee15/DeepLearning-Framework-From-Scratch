#pragma once

#include <vector>
#include "core/Tensor.h"
#include "core/Layer.h"

class Activation;

using namespace std;

class DenseLayer : public Layer {
    private:

        // Constants
        static const double HE_INT_GAIN;

        // Instance Variables
        Tensor activations;
        Tensor preActivations;
        Tensor weights;
        Tensor dZ;
        Activation *activation;
        vector<double> biases;


        // Methods
        void initWeights(size_t, size_t);
        vector<uint32_t> generateThreadSeeds() const;

    public:
        // Constructor
        DenseLayer(size_t, size_t, Activation*);

        // Methods
        void calActivations(const Tensor&) override;
        const Tensor getActivations() const override;
        Tensor getOutputGradient() const override;
        void backprop(const Tensor&, double, const Tensor&, bool) override;
        ~DenseLayer();
        void writeBin(ofstream&) const override;
        void loadWeightsAndBiases(ifstream&) override;
        uint32_t getEncoding() const override;
};