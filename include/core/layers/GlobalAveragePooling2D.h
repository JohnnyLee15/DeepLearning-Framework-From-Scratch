#pragma once

#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"
#include <string>

class GlobalAveragePooling2D : public Layer {
    private:

        // Instance Variables
        Tensor output;
        Tensor dX;

        void reShapeBatch(size_t);
        void writeBinInternal(ofstream&) const override;
        void checkBuildSize(const vector<size_t>&) const;

    public:
        // Methods
        void build(const vector<size_t>&, bool isInference = false) override;

        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;

        const Tensor& getOutput() const override;
        Tensor& getOutputGradient() override;

        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;
        Layer::Encodings getEncoding() const override;

        Layer* clone() const override;

        // GPU Interface
        #ifdef __APPLE__
            void forwardGpu(const Tensor&, GpuCommandBuffer) override;
            void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer) override;
        #endif
};