#pragma once

#include "losses/Loss.h"

class MSE : public Loss {
    public:
        // Methods
        double calculateTotalLoss(const vector<double>&, const Matrix&) const override;    
        Matrix calculateGradient(const vector<double>&, const Matrix&) const override;
        double formatLoss(double) const override;
};