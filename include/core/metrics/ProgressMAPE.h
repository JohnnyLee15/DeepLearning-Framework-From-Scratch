#pragma once

#include "core/metrics/ProgressMetric.h"

class ProgressMAPE : public ProgressMetric {
    private:
        
        // Constants
        static const string NAME;

        // Instance Variables
        size_t numNonZeroTargets;
        float runningSum;

        // Methods
        void accumulateMAPE(
            const Tensor&, const vector<float>&, const Loss*, const Tensor&, float
        );
        
    public:
        // Methods
        void init(size_t) override;
        string getName() const override;
        void update(const Batch&, const Loss*, const Tensor&, float) override;
        void update(
            const Tensor&, const vector<float>&, const Loss*, const Tensor&, float
        ) override;
        float calculate() const override;
};