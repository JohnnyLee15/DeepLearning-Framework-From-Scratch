#pragma once

#include "Activation.h"

class Softmax : public Activation {
    public:
        double getMaxPreActivation(const vector<double>&) const;
        static const double SOFTMAX_BIAS;

    public:
        vector<double> activate(const vector<double>&) const override;
        vector<double> calculateGradient(const vector<double>&) const override;
        double initBias() const override;
};