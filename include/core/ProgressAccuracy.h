#pragma once
#include "core/ProgressMetric.h"
#include <vector>

class ProgressAccuracy : public ProgressMetric {
    private:
        static const string NAME;
        
        size_t correctPredictions;
        vector<double> predictions;

    public:
        ProgressAccuracy(size_t);
        string getName() const override;
        void init() override;
        void update(const Batch&, const Loss*, const Tensor&, double) override;
        double calculate() const override;
};