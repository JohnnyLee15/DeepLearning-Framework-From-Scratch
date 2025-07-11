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
        size_t numNeurons;
        Tensor activations;
        Tensor preActivations;
        Tensor weights;
        Tensor dZ;
        Activation *activation;
        vector<double> biases;

        // Methods
        void initWeights();
        vector<uint32_t> generateThreadSeeds() const;
        void loadActivation(ifstream&);
        void checkBuildSize(const vector<size_t>&) const;

    public:
        // Constructors
        DenseLayer(size_t, Activation*);
        DenseLayer();

        // Methods
        void forward(const Tensor&) override;
        const Tensor& getOutput() const override;
        Tensor getOutputGradient() const override;
        void backprop(const Tensor&, double, const Tensor&, bool) override;
        ~DenseLayer();
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
        void build(const vector<size_t>&) override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
};