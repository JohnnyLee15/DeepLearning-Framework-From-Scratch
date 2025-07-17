#pragma once

#include <vector>
#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"

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
        bool isLoadedDense;
        

        // Methods
        void initWeights();
        void initParams(size_t);
        
        void ensureGpu();
        void ensureCpu();

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

        #ifdef __APPLE__
            void forwardGpu(const Tensor&, GpuCommandBuffer) override;
            void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer) override;
            void writeToBinGpu();
        #endif
};