#pragma once

#include "core/model/NeuralNet.h"
#include <string>

using namespace std;

class EarlyStop {
    private:
        // Instance Variables
        size_t patience;
        float minDelta;
        size_t warmupEpochs;

        size_t badEpochs;
        float bestLoss;

        string bestWeightsPath;

        // Methods
        void saveBestWeights(const NeuralNet&);

    public:
        // Constructors
        EarlyStop(size_t patience = 5, float minDelta = 1e-4f, size_t warmupEpochs = 5);
        bool shouldStop(float, size_t, const NeuralNet&);
        bool hasBestWeights() const;
        const string& getBestWeightPath() const;
        void deleteBestWeights();
};