#pragma once

#include "Activation.h"

class Softmax : public Activation {
    public:
        // Constants
        static const double SOFTMAX_BIAS;

        // Methods
        double getMaxPreActivation(const vector<double>&, size_t, size_t) const;
        void activateRow(vector<double>&, const vector<double>&, size_t, size_t) const;


    public:
        // Methods
        Matrix activate(const Matrix&) const override;
        Matrix calculateGradient(const Matrix&) const override;
        vector<double> initBias(size_t) const override;
};