#pragma once

#include "losses/Loss.h"

class SoftmaxCrossEntropy : public Loss {
    private:
        // Constants
        static const float CROSS_ENTROPY_EPSILON;
        // Methods
        float calculateDerivative(float, size_t, size_t) const;

    public:
        // Methods
        Tensor calculateGradient(const vector<float>&, const Tensor&) const override;
        float calculateTotalLoss(const vector<float>&, const Tensor&) const override;
        uint32_t getEncoding() const override;

        // Static Methods
        static void checkInvalidGradientCall();
};