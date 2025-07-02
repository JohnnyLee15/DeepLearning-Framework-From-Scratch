#pragma once
#include "core/Task.h"
#include <unordered_map>

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
        void createLabelMap(const vector<string>&);
        double processBatch(Batch&, vector<double>&, const Tensor&, const Loss*) const override;
        double calculateProgressMetric(const Batch&, const Tensor&, const vector<double>&, EpochStats&) const override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
};