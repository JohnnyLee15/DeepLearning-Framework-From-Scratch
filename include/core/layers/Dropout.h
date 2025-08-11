#pragma once

#include <vector>
#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"


class Dropout : public Layer {
    private:
        // Instance Variables
        float rate;

        Tensor mask;
        Tensor output;
        Tensor dX;

        // Methods
        vector<uint32_t> generateThreadSeeds() const;
        void generateMask();
        void writeBinInternal(ofstream&) const override;
        void reShapeBatch(size_t);

    public:
        // Constructors
        Dropout(float);
        Dropout();

        // Methods
        void build(const vector<size_t>&, bool isInference = false) override;

        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;

        const Tensor& getOutput() const override;
        Tensor& getOutputGradient() override;

        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        Layer::Encodings getEncoding() const override;

        void loadFromBin(ifstream&) override;

        Layer* clone() const override;

        // GPU Interface
        #ifdef __APPLE__
            void forwardGpu(const Tensor&, GpuCommandBuffer) override;
            void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer) override;
        #endif
};