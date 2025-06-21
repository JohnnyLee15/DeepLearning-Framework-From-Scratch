#pragma once

#include "Activation.h"

class ReLU : public Activation {
    private:
        // Constants
        static const double RELU_BIAS;
    
    public:
        // Methods
        Matrix activate(const Matrix&) const override;
        Matrix calculateGradient(const Matrix&) const override;
        vector<double> initBias(size_t) const override;
        uint32_t getEncoding() const override;
};