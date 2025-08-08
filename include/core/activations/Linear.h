#pragma once

#include "core/activations/Activation.h"

class Linear : public Activation {
    private:
        // Constants
        static const float LINEAR_BIAS;
    
    public:
        // Methods
        Tensor initBias(size_t) const override;

        void activate(const Tensor&, Tensor&)  const override;
        void calculateGradient(const Tensor&, Tensor&) const override;
        
        Activation::Encodings getEncoding() const override;

        Activation* clone() const override;

        // GPU Interface
        #ifdef __APPLE__
            void activateGpu(const Tensor&, Tensor&, GpuCommandBuffer) const override;
            void calculateGradientGpu(const Tensor&, Tensor&, GpuCommandBuffer) const override;
        #endif
};