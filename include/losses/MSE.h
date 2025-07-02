#pragma once

#include "losses/Loss.h"

class MSE : public Loss {
    public:
        // Methods
        double calculateTotalLoss(const vector<double>&, const Tensor&) const override;    
        Tensor calculateGradient(const vector<double>&, const Tensor&) const override;
        double formatLoss(double) const override;
        uint32_t getEncoding() const override;
};