#pragma once

#include "Activation.h"

class Linear : public Activation {
    private:
        // Constants
        static const double LINEAR_BIAS;
    
    public:
        // Methods
        Tensor activate(const Tensor&) const override;
        Tensor calculateGradient(const Tensor&) const override;
        vector<double> initBias(size_t) const override;
        uint32_t getEncoding() const override;
};