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

        static const size_t GPU_FAST;
        static const size_t GPU_NAIVE;
        static const size_t CPU;

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

        size_t executionMode;
        
        // Methods
        void initBiases();
        void initWeights(size_t);
        void initParams(size_t, bool);
        void initExecutionMode();
        void checkBuildSize(const vector<size_t>&) const;
        
        void ensureGpu();
        void syncBuffers() override;

        vector<uint32_t> generateThreadSeeds() const;
        void loadActivation(ifstream&);
        void writeBinInternal(ofstream&) const override;

        void reShapeBatch(size_t);
        
    public:
        // Constructors
        Dense(size_t, Activation*);
        Dense();
        Dense(const Dense&);

        // Destructor
        ~Dense();

        // Methods
        void build(const vector<size_t>&, bool isInference = false) override;
        void allocateGradientBuffers(size_t, bool);
        void allocateForwardBuffers();
        void deallocateGradientBuffers(bool);

        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;

        const Tensor& getOutput() const override;
        Tensor& getOutputGradient() override;
        
        Layer::Encodings getEncoding() const override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;

        void loadFromBin(ifstream&) override;

        Layer* clone() const override;

        const Tensor& getWeights() const override;
        const Tensor& getBiases() const override;
        const Tensor& getDeltaInputs() const override;

        // GPU Interface
        #ifdef __APPLE__
            void forwardGpu(const Tensor&, GpuCommandBuffer) override;
            void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer) override;
            void backpropGpuNaive(const Tensor&, float, Tensor&, bool, GpuCommandBuffer);
            void backpropGpuFast(const Tensor&, float, Tensor&, bool, GpuCommandBuffer);
            void downloadOutputFromGpu() override;
        #endif
};