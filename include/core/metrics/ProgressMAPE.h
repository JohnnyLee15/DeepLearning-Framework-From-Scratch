#pragma once

#include "core/metrics/ProgressMetric.h"

class ProgressMAPE : public ProgressMetric {
    private:
        
        // Constants
        static const string NAME;

        // Instance Variables
        size_t numNonZeroTargets;
        float runningSum;
        
    public:

        // Constructor
        ProgressMAPE(size_t);

        // Methods
        void init() override;
        string getName() const override;
        void update(const Batch&, const Loss*, const Tensor&, float) override;
        float calculate() const override;
};