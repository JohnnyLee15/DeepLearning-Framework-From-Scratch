#pragma once

#include "Activation.h"

class Relu : public Activation {
    private:
        static const double RELU_BIAS;
    
    public:
        Matrix activate(const Matrix&) const override;
        Matrix calculateGradient(const Matrix&) const override;
        vector<double> initBias(size_t) const override;
};