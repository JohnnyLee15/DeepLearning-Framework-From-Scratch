#pragma once

#include "core/losses/Loss.h"

class MSE : public Loss {
    public:
        // Methods
        float calculateTotalLoss(const Tensor&, const Tensor&) const override;    
        void calculateGradient(const Tensor&, const Tensor&, Tensor&) const override;

        float formatLoss(float) const override;

        uint32_t getEncoding() const override;

        Loss* clone() const override;

        // GPU Interface
        #ifdef __APPLE__
            void calculateGradientGpu(const Tensor&, const Tensor&, Tensor&, GpuCommandBuffer) const override;
        #endif
};