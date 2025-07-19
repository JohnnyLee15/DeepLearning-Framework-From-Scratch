#pragma once

#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"
#include <string>

class MaxPooling2D : public Layer {
    private:

        // Instance Variables
        size_t kRows;
        size_t kCols;
        size_t stride;

        Tensor dZ;

        Tensor::Paddings padding;
        Tensor pooledOutput;
        vector<size_t> maxIndices;
        
        // Methods
        void initStride(size_t);

    public:
        // Constructor
        MaxPooling2D(size_t, size_t, size_t, const string&);

        // Methods
        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;

        const Tensor& getOutput() const override;
        Tensor& getOutputGradient() override;

        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        Layer::Encodings getEncoding() const override;

        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
};