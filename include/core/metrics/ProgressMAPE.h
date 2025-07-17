#pragma once
#include "core/metrics/ProgressMetric.h"

class ProgressMAPE : public ProgressMetric {
    private:
        static const string NAME;

        size_t numNonZeroTargets;
        float runningSum;

        float computeMAPE(const vector<float>&, const vector<float>&);
        
    public:
        ProgressMAPE(size_t);
        string getName() const override;
        void init() override;
        void update(const Batch&, const Loss*, const Tensor&, float) override;
        float calculate() const override;
};