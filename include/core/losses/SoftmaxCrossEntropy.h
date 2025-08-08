#pragma once

#include "core/losses/Loss.h"

class SoftmaxCrossEntropy : public Loss {
    private:

        // Constants
        static const float CROSS_ENTROPY_EPSILON;

        // Methods
        float calculateDerivative(float, size_t, size_t) const;

    public:

        // Static Methods
        static void checkInvalidGradientCall();

        // Methods
        void calculateGradient(const Tensor&, const Tensor&, Tensor&) const override;
        float calculateTotalLoss(const Tensor&, const Tensor&) const override;

        uint32_t getEncoding() const override;

        Loss* clone() const override;

        // GPU Interface
        #ifdef __APPLE__
            void calculateGradientGpu(const Tensor&, const Tensor&, Tensor&, GpuCommandBuffer) const override;
        #endif
};