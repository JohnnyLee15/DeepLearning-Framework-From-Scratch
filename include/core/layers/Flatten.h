#pragma once

#include <vector>
#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"

class Flatten : public Layer {
    private:

        // Instance Variables
        vector<size_t> inShape;
        vector<size_t> outShape;

        Tensor output;
        Tensor dX;

        // Methods
        void checkInputSize(const vector<size_t>&) const;

    public:
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