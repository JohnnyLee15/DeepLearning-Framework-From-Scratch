#pragma once

#include "Activation.h"

class Softmax : public Activation {
    public:
        double getMaxPreActivation(const vector<double>&, size_t, size_t) const;
        void activateRow(vector<double>&, const vector<double>&, size_t, size_t) const;
        static const double SOFTMAX_BIAS;

    public:
        Matrix activate(const Matrix&) const override;
        Matrix calculateGradient(const Matrix&) const override;
        vector<double> initBias(size_t) const override;
};