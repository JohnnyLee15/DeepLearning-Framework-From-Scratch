#pragma once
#include <string>
#include <chrono>

class Batch;
class Loss;
class Tensor;


using namespace std;

class ProgressMetric {
    private:
        double totalLoss;
        double avgLoss;
        double timeElapsed;
        size_t numSamples;
        std::chrono::steady_clock::time_point startTime;
        string progressMetricName;
        size_t samplesProcessed;

    public:
        ProgressMetric(size_t);

        virtual string getName() const = 0;
        virtual void init();
        virtual void update(const Batch&, const Loss*, const Tensor&, double);
        virtual double calculate() const = 0;
        size_t getSamplesProcessed() const;
        double getTotalLoss() const;
        size_t getNumSamples() const;
        double getTimeElapsed() const;
        double getAvgLoss() const;

        virtual ~ProgressMetric() = default;
};