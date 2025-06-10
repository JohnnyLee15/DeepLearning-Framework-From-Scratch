#pragma once
#include "core/Task.h"

class Matrix;

class RegressionTask : public Task {
    private:
        // Constants
        static const string PROGRESS_METRIC_NAME;

        // Instance Variables
        Scalar *targetScalar;

        // Methods
        double computeMAPE(const vector<double>&, const vector<double>&, EpochStats&) const;

    public:
        // Constructor
        RegressionTask();

        // Methods
        vector<double> getTarget(const vector<string>&) override;
        vector<double> parsePredictions(const Matrix&) const override;
        Matrix predict(const Matrix&) const override;
        void resetToRaw() override;
        void setTargetScalar(Scalar*) override;
        void fitScalars(Matrix&, vector<double>&, Matrix&, vector<double>&) override;
        virtual double processBatch(Batch&, vector<double>&, const Matrix&, const Loss*) const override;
        virtual double calculateProgressMetric(const Batch&, const Matrix&, const vector<double>&, EpochStats&) const override;
        ~RegressionTask();

};