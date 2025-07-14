#pragma once

#include "Activation.h"

class Softmax : public Activation {
    public:
        // Constants
        static const float SOFTMAX_BIAS;

        // Methods
        float getMaxPreActivation(const vector<float>&, size_t, size_t) const;
        void activateRow(vector<float>&, const vector<float>&, size_t, size_t) const;


    public:
        // Methods
        void activate(const Tensor&, Tensor&)  const override;
        void calculateGradient(const Tensor&, Tensor&) const override;
        Tensor initBias(size_t) const override;
        bool isFused() const override;
        uint32_t getEncoding() const override;

        // Gpu
        #ifdef __OBJC__
            void calculateGradientGpu(const Tensor&, Tensor&) const override;
        #endif
};