#pragma once

#include <vector>
#include "core/Tensor.h"
#include "core/Layer.h"

class Activation;

using namespace std;

class DenseLayer : public Layer {
    private:

        // Constants
        static const float HE_INT_GAIN;

        // Instance Variables
        size_t numNeurons;
        Tensor activations;
        Tensor preActivations;
        Tensor weights;
        Tensor dB;
        Tensor dW;
        Tensor dX;
        Tensor dA;
        Tensor biases;
        Activation *activation;
        

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
        Tensor& getOutput() override;
        Tensor& getOutputGradient() override;
        void backprop(const Tensor&, float, Tensor&, bool) override;
        ~DenseLayer();
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
        void build(const vector<size_t>&) override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        void reShapeBatch(size_t);
        void revertBatchReShape();
        void downloadOutputFromGpu() override;

        #ifdef __OBJC__
            void forwardGpu(const Tensor&) = 0;
        #endif
};