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
        // Instance Variables
        size_t numKernals;
        size_t kRows;
        size_t kCols;

        Tensor paddedInput;
        Tensor kernals;
        Tensor activations;
        Tensor preActivations;
        Tensor dB;
        Tensor dW;
        Tensor dA;
        Tensor dX;
        Tensor gradBuf;
        Tensor biases;

        WindowDims winIn;
        WindowDims winGrad;
        Activation *activation;
        Tensor::Paddings padding;
        size_t stride;
        bool isLoadedConv2D;

        // Methods
        void initKernals();
        void initGradBuf();
        void initStride(size_t);
        void initParams(size_t);
        void checkBuildSize(const vector<size_t>&) const;

        void ensureGpu();

        vector<uint32_t> generateThreadSeeds() const;

        void reShapeBatch(size_t);

    public:
        // Constructor
        Conv2D(size_t, size_t, size_t, size_t, const string&, Activation*);

        // Methods
        void build(const vector<size_t>&) override;

        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;

        const Tensor& getOutput() const override;
        Tensor& getOutputGradient() override;
        
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        Layer::Encodings getEncoding() const override;

        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
};