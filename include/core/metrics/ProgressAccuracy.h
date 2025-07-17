#pragma once
#include "core/metrics/ProgressMetric.h"
#include <vector>

class ProgressAccuracy : public ProgressMetric {
    private:
        static const string NAME;
        
        size_t correctPredictions;
        vector<float> predictions;

    public:
        ProgressAccuracy(size_t);
        string getName() const override;
        void init() override;
        void update(const Batch&, const Loss*, const Tensor&, float) override;
        float calculate() const override;
};