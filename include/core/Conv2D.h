#pragma once
#include "core/Tensor.h"
#include "core/Layer.h"
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
        Tensor kernals;
        Tensor activations;
        Tensor preActivations;
        Tensor dZ;
        vector<double> biases;
        Activation *activation;
        Tensor::Paddings padding;
        size_t stride;

        // Methods
        void initKernals();
        void initStride(size_t);
        void checkBuildSize(const vector<size_t>&) const;

        vector<uint32_t> generateThreadSeeds() const;

    public:
        Conv2D(size_t, size_t, size_t, size_t, const string&, Activation*);
        void forward(const Tensor&) override;
        void backprop(const Tensor&, double, const Tensor&, bool) override;
        const Tensor& getOutput() const override;
        Tensor getOutputGradient() const override;
        void build(const vector<size_t>&) override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
};