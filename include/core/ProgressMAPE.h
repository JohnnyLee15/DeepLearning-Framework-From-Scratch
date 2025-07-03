#pragma once
#include "core/ProgressMetric.h"

class ProgressMAPE : public ProgressMetric {
    private:
        static const string NAME;

        size_t numNonZeroTargets;
        double runningSum;

        double computeMAPE(const vector<double>&, const vector<double>&);
        
    public:
        ProgressMAPE(size_t);
        string getName() const override;
        void init() override;
        void update(const Batch&, const Loss*, const Tensor&, double) override;
        double calculate() const override;
};