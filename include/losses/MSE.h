#pragma once

#include "losses/Loss.h"

class MSE : public Loss {
    public:
        // Methods
        float calculateTotalLoss(const vector<float>&, const Tensor&) const override;    
        Tensor calculateGradient(const vector<float>&, const Tensor&) const override;
        float formatLoss(float) const override;
        uint32_t getEncoding() const override;
};