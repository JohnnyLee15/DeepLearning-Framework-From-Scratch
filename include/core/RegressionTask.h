#pragma once
#include "core/Task.h"

class RegressionTask : public Task {
    private:
        // Constants
        static const string PROGRESS_METRIC_NAME;

        // Instance Variables
        Scalar *targetScalar;

        // Methods
        double computeMAPE(const vector<double>&, const vector<double>&, EpochStats&) const;
        void checkNumOutputNeurons(size_t) const;

    public:
        // Constructor
        RegressionTask();

        // Methods
        vector<double> getTarget(const vector<string>&) override;
        Tensor predict(const Tensor&) const override;
        void resetToRaw() override;
        void setTargetScalar(Scalar*) override;
        void fitScalars(Tensor&, vector<double>&) override;
        void transformScalars(Tensor&, vector<double>&) override;
        void reverseTransformScalars(Tensor&, vector<double>&) override;
        double processBatch(Batch&, vector<double>&, const Tensor&, const Loss*) const override;
        double calculateProgressMetric(const Batch&, const Tensor&, const vector<double>&, EpochStats&) const override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
        ~RegressionTask();

};