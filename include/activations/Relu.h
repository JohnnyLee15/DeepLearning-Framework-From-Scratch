#pragma once

#include "Activation.h"

class Relu : public Activation {
    private:
        static const double RELU_BIAS;
    
    public:
        vector<double> activate(const vector<double>&) const override;
        vector<double> calculateGradient(const vector<double>&) const override;
        double initBias() const override;
};