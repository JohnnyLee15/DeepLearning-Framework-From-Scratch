#pragma once

#include "losses/Loss.h"

class SoftmaxCrossEntropy : public Loss {
    private:
        // Constants
        static const double CROSS_ENTROPY_EPSILON;
        // Methods
        double calculateDerivative(double, size_t, size_t) const;

    public:
        // Methods
        Matrix calculateGradient(const vector<double>&, const Matrix&) const override;
        double calculateTotalLoss(const vector<double>&, const Matrix&) const override;

        // Static Methods
        static void checkInvalidGradientCall();
};