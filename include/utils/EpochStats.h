#pragma once
#include <string>
#include <chrono>

using namespace std;

// Structs
struct EpochStats {
    double totalLoss;
    double timeElapsed;
    double avgLoss;
    double mapeSum;
    double progressMetric;
    size_t samplesProcessed;
    size_t correctPredictions; // For classification tasks
    size_t numSamples;
    size_t nonZeroTargets; // For regression tasks
    std::chrono::steady_clock::time_point startTime;
    string progressMetricName;
};