#pragma once

#include <cstdint>
#include <vector>
#include "core/gpu/GpuTypes.h"

class Tensor;

using namespace std;

class Activation {
    public:

        // Enums
        enum Encodings : uint32_t {
            Linear,
            ReLU,
            Softmax
        };

        // Virtual Destructor
        virtual ~Activation() = default;

        // Methods
        virtual Tensor initBias(size_t) const = 0;

        virtual void activate(const Tensor&, Tensor&) const = 0;
        virtual void calculateGradient(const Tensor&, Tensor&) const = 0;

        virtual bool isFused() const;
        virtual Encodings getEncoding() const = 0;
        
        // GPU Interface
        #ifdef __APPLE__
            virtual void activateGpu(const Tensor&, Tensor&, GpuCommandBuffer) const = 0;
            virtual void calculateGradientGpu(const Tensor&, Tensor&, GpuCommandBuffer) const = 0;
        #endif
};