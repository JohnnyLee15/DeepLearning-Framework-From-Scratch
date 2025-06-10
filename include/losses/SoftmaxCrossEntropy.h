#pragma once

#include "losses/CrossEntropy.h"

class SoftmaxCrossEntropy : public CrossEntropy{
    private:
        // Methods
        double calculateDerivative(double, size_t, size_t) const;

    public:
        // Methds
        Matrix calculateGradient(const vector<double>&, const Matrix&) const override;
        bool isFused() const override;
};