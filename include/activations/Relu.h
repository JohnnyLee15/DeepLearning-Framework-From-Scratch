#pragma once

#include "Activation.h"

class ReLU : public Activation {
    private:
        // Constants
        static const float RELU_BIAS;
    
    public:
        // Methods
        void activate(const Tensor&, Tensor&)  const override;
        void calculateGradient(const Tensor&, Tensor&) const override;
        Tensor initBias(size_t) const override;
        uint32_t getEncoding() const override;

        // Gpu
        #ifdef __OBJC__
            void calculateGradientGpu(const Tensor&, Tensor&) const override;
        #endif
};