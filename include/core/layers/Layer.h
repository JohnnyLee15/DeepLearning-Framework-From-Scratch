#pragma once

#include <fstream>
#include <cstdint>
#include "core/gpu/GpuTypes.h"

class Tensor;

using namespace std;

class Layer {
    private:
        // Instance Variables
        size_t maxBatchSize;
        bool isMaxBatchSet;

    public:

        // Enums
        enum Encodings : uint32_t {
            Dense,
            Conv2D,
            MaxPooling2D,
            Flatten,
            None
        };
        // Constructor
        Layer();

        // Virtual Destructor
        virtual ~Layer() = default;

        // Methods
        virtual void build(const vector<size_t>&) = 0;

        virtual void forward(const Tensor&) = 0;
        virtual void backprop(const Tensor&, float, Tensor&, bool) = 0;

        virtual const Tensor& getOutput() const = 0;
        virtual Tensor& getOutputGradient() = 0;

        virtual vector<size_t> getBuildOutShape(const vector<size_t>&) const = 0;
        virtual Encodings getEncoding() const = 0;
        size_t getMaxBatchSize() const;
        
        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&) = 0;

        // GPU Interface
        #ifdef __APPLE__
            virtual void forwardGpu(const Tensor&, GpuCommandBuffer);
            virtual void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer);
            virtual void downloadOutputFromGpu();
        #endif
};