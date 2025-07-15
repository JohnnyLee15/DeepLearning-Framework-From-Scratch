#pragma once

#include "core/Layer.h"
#include <vector>
#include "core/Tensor.h"

class Flatten : public Layer {
    private:
        vector<size_t> inShape;
        vector<size_t> outShape;
        Tensor output;
        Tensor dZ;

        void checkInputSize(const vector<size_t>&) const;

    public:
        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;
        Tensor& getOutput() override;
        Tensor& getOutputGradient() override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        void build(const vector<size_t>&) override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
};