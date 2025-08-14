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
        // Methods
        void init(size_t) override;
        string getName() const override;
        void updateCorrectPredictions(
            const Tensor&, const vector<float>&, const Tensor&, 
            const vector<size_t> *indices = nullptr
        );
        void update(const Batch&, const Loss*, const Tensor&, float) override;
        void update(
            const Tensor&, const vector<float>&, const Loss*, const Tensor&, float
        ) override;
        float calculate() const override;
};