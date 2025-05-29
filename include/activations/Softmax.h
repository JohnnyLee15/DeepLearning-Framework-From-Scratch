#pragma once

#include "Activation.h"

class Softmax : public Activation {
    public:
        double getMaxPreActivation(const Matrix&, int) const;
        void activateRow(Matrix&, const Matrix&, int) const;
        static const double SOFTMAX_BIAS;

    public:
        Matrix activate(const Matrix&) const override;
        Matrix calculateGradient(const Matrix&) const override;
        vector<double> initBias(size_t) const override;
};