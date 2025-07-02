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
        Tensor activate(const Tensor&) const override;
        Tensor calculateGradient(const Tensor&) const override;
        vector<double> initBias(size_t) const override;
        bool isFused() const override;
        uint32_t getEncoding() const override;
};