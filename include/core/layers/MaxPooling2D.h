#pragma once

#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"
#include <string>
#include "core/gpu/MetalBuffer.h"

class MaxPooling2D : public Layer {
    private:

        // Instance Variables
        size_t kRows;
        size_t kCols;
        size_t stride;

        Tensor paddedInput;
        Tensor dX;

        WindowDims winIn;
        Tensor::Paddings padding;
        Tensor pooledOutput;
        vector<size_t> maxIndices;

        // GPU Instance Variables
        #ifdef __APPLE__
            MetalBuffer maxIndicesGpu;
        #endif
        
        // Methods
        void initStride(size_t);
        void initMaxIndices();
        void checkBuildSize(const vector<size_t>&) const;

        void reShapeBatch(size_t);

    public:
        // Constructor
        MaxPooling2D(size_t, size_t, size_t, const string&);

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

        // GPU Interface
        #ifdef __APPLE__
            void forwardGpu(const Tensor&, GpuCommandBuffer) override;
            // void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer) override;
        #endif
};