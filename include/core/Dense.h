#pragma once

#include <vector>
#include "core/Tensor.h"
#include "core/Layer.h"

class Activation;

using namespace std;

class Dense : public Layer {
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
        Dense(size_t, Activation*);
        Dense();

        // Methods
        void forward(const Tensor&) override;
        Tensor& getOutput() override;
        Tensor& getOutputGradient() override;
        void backprop(const Tensor&, float, Tensor&, bool) override;
        ~Dense();
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
        void build(const vector<size_t>&) override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        void reShapeBatch(size_t);
        void ensureGpu();

        #ifdef __OBJC__
            void forwardGpu(const Tensor&, id<MTLCommandBuffer>) override;
            void backpropGpu(const Tensor&, float, Tensor&, bool, id<MTLCommandBuffer>) override;
            void downloadOutputFromGpu() override;
        #endif
};