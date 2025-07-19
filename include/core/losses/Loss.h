#pragma once
#include <cstdint>
#include <vector>
#include "core/gpu/GpuTypes.h"

class Tensor;

using namespace std;

class Loss {
    public:
        // Enums
        enum Encodings : uint32_t {
            MSE,
            SoftmaxCrossEntropy
        };

        // Virtual Destructor
        virtual ~Loss() = default;

        // Methods
        virtual float calculateTotalLoss(const Tensor&, const Tensor&) const = 0;    
        virtual void calculateGradient(const Tensor&, const Tensor&, Tensor&) const = 0;
        
        virtual float formatLoss(float) const;
        virtual uint32_t getEncoding() const = 0;

        // GPU Interface
        #ifdef __APPLE__
             virtual void calculateGradientGpu(const Tensor&, const Tensor&, Tensor&, GpuCommandBuffer) const = 0;
        #endif
};