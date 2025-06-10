#pragma once

#include "Activation.h"

class Linear : public Activation {
    private:
        // Constants
        static const double LINEAR_BIAS;
    
    public:
        // Methods
        Matrix activate(const Matrix&) const override;
        Matrix calculateGradient(const Matrix&) const override;
        vector<double> initBias(size_t) const override;
};