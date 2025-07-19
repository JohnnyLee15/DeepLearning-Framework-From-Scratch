#pragma once

#include "core/metrics/ProgressMetric.h"
#include <vector>

class ProgressAccuracy : public ProgressMetric {
    private:

        // Constants
        static const string NAME;
        
        // Instance Variables
        size_t correctPredictions;
        vector<float> predictions;

    public:
        // Constructor
        ProgressAccuracy(size_t);

        // Methods
        void init() override;
        string getName() const override;
        void update(const Batch&, const Loss*, const Tensor&, float) override;
        float calculate() const override;
};