#pragma once

#include "core/model/Pipeline.h"
#include <string>

using namespace std;

class EarlyStop {
    private:
        // Instance Variables
        size_t patience;
        float minDelta;
        size_t warmupEpochs;
        Pipeline *pipe;

        size_t badEpochs;
        float bestLoss;

        string bestPipePath;

        // Methods
        void saveBestPipe();

    public:
        // Constructors
        EarlyStop();
        EarlyStop(
            Pipeline*, size_t patience = 5, float minDelta = 1e-4f, 
            size_t warmupEpochs = 5
        );

        // Methods
        bool isEnabled() const;

        bool shouldStop(float, size_t);
};