#pragma once
#include "core/Task.h"
#include <unordered_map>

class Matrix;

class ClassificationTask : public Task {
    private:
        // Constants
        static const string PROGRESS_METRIC_NAME;

        // Instance Variables
        unordered_map<string, int> labelMap;

    public:
        // Constructor
        ClassificationTask();

        // Methods
        vector<double> getTarget(const vector<string>&) override;
        vector<double> parsePredictions(const Matrix&) const override;
        void createLabelMap(const vector<string>&);
        virtual double processBatch(Batch&, vector<double>&, const Matrix&, const Loss*) const override;
        virtual double calculateProgressMetric(const Batch&, const Matrix&, const vector<double>&, EpochStats&) const override;
};