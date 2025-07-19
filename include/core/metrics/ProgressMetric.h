#pragma once

#include <string>
#include <chrono>

class Batch;
class Loss;
class Tensor;


using namespace std;

class ProgressMetric {
    private:

        // Instance Variables
        float totalLoss;
        float avgLoss;
        float timeElapsed;
        size_t numSamples;
        std::chrono::steady_clock::time_point startTime;
        string progressMetricName;
        size_t samplesProcessed;

    public:
        
        // Constructor
        ProgressMetric(size_t);

        // Virtual Destructor
        virtual ~ProgressMetric() = default;

        // Methods
        virtual void init();
        
        virtual string getName() const = 0;
        size_t getSamplesProcessed() const;
        float getTotalLoss() const;
        size_t getNumSamples() const;
        float getTimeElapsed() const;
        float getAvgLoss() const;

        virtual void update(const Batch&, const Loss*, const Tensor&, float);
        virtual float calculate() const = 0;
};