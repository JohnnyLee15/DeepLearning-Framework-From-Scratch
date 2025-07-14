#pragma once
#include <string>
#include <chrono>

class Batch;
class Loss;
class Tensor;


using namespace std;

class ProgressMetric {
    private:
        float totalLoss;
        float avgLoss;
        float timeElapsed;
        size_t numSamples;
        std::chrono::steady_clock::time_point startTime;
        string progressMetricName;
        size_t samplesProcessed;

    public:
        ProgressMetric(size_t);

        virtual string getName() const = 0;
        virtual void init();
        virtual void update(const Batch&, const Loss*, const Tensor&, float);
        virtual float calculate() const = 0;
        size_t getSamplesProcessed() const;
        float getTotalLoss() const;
        size_t getNumSamples() const;
        float getTimeElapsed() const;
        float getAvgLoss() const;

        virtual ~ProgressMetric() = default;
};