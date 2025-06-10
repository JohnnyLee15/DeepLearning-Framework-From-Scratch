#pragma once
#include <string>
#include <chrono>

using namespace std;

// Struct
struct EpochStats {
    double totalLoss;
    double timeElapsed;
    double avgLoss;
    double mapeSum;
    double progressMetric;
    size_t samplesProcessed;
    size_t correctPredictions;
    size_t numSamples;
    std::chrono::steady_clock::time_point startTime;
    string progressMetricName;
};