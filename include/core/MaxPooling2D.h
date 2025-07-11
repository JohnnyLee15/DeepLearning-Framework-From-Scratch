#pragma once

#include "core/Layer.h"
#include "core/Tensor.h"
#include <string>

class MaxPooling2D : public Layer {
    private:
        // Instance Variables
        size_t kRows;
        size_t kCols;
        size_t stride;
        Tensor::Paddings padding;
        Tensor pooledOutput;
        vector<size_t> maxIndices;
        Tensor dZ;

        // Methods
        void initStride(size_t);

    public:
        MaxPooling2D(size_t, size_t, size_t, const string&);
        void forward(const Tensor&) override;
        void backprop(const Tensor&, double, const Tensor&, bool) override;
        const Tensor& getOutput() const override;
        Tensor getOutputGradient() const override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
};