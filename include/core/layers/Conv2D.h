#pragma once

#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"
#include <vector>
#include <cstdint>
#include <string>

class Activation;

using namespace std;

class Conv2D : public Layer {
    private:
        // Constants
        static const float HE_INT_GAIN;
        
        static const size_t GPU_FAST;
        static const size_t GPU_NAIVE;
        static const size_t CPU;

        // Instance Variables
        size_t numKernels;
        size_t kRows;
        size_t kCols;

        Tensor paddedInput;
        Tensor im2ColInBuf;
        Tensor kernels;
        Tensor fastKernels;
        Tensor activations;
        Tensor preActivations;
        vector<size_t> im2ColPreActShape;
        vector<size_t> preActTensorShape;
        Tensor dB;
        Tensor dW;
        Tensor dA;
        Tensor dX;
        Tensor gradIm2ColBuf;
        Tensor gradBuf;
        Tensor biases;

        WindowDims winIn;
        WindowDims winGrad;
        Activation *activation;
        Tensor::Paddings padding;
        size_t stride;
        size_t executionMode;

        // Methods
        void initKernels(size_t);
        void initBiases();
        void initGradBuf(bool);
        void initStride(size_t);
        void initParams(size_t);
        void initExecutionMode(size_t, size_t);

        void flattenKernels();
        void unflattenKernels();

        void allocateGradientBuffers(size_t, size_t, size_t, bool);
        void allocateForwardBuffers(size_t, size_t, size_t);
        void deallocateGradientBuffers(bool);
        
        void checkBuildSize(const vector<size_t>&) const;

        void ensureGpu();
        void syncBuffers() override;

        vector<uint32_t> generateThreadSeeds() const;
        void loadActivation(ifstream&);
        void writeBinInternal(ofstream&) const override;

        void reShapeBatch(size_t);
        void reShapeGpuFastBuffers(size_t, size_t);
        void reShapeCpuBuffers(size_t);

    public:
        // Constructors
        Conv2D(size_t, size_t, size_t, size_t, const string&, Activation*);
        Conv2D();
        Conv2D(const Conv2D&);

        // Methods
        void build(const vector<size_t>&, bool isInference = false) override;

        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;

        const Tensor& getOutput() const override;
        Tensor& getOutputGradient() override;
        
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        Layer::Encodings getEncoding() const override;

        void loadFromBin(ifstream&) override;

        const Tensor& getWeights() const override;
        const Tensor& getBiases() const override;
        const Tensor& getDeltaInputs() const override;

        Layer* clone() const override;

        // GPU Interface
        #ifdef __APPLE__
            void forwardGpu(const Tensor&, GpuCommandBuffer) override;
            void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer) override;
            void backpropGpuNaive(const Tensor&, float, Tensor&, bool, GpuCommandBuffer);
            void backpropGpuFast(const Tensor&, float, Tensor&, bool, GpuCommandBuffer);
        #endif
};