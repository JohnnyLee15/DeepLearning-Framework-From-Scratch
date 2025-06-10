#pragma once
#include <string>
#include <vector>
#include "utils/EpochStats.h"

class Batch;
class Layer;
class Matrix;
class Loss;
class Scalar;

using namespace std;

class Task {
    private:
        // Instance Variables
        string progressMetricName;
        Scalar *featureScalar;
        
    public:
        // Constructor
        Task(string);

        // Methods
        virtual vector<double> getTarget(const vector<string>&) = 0;
        virtual vector<double> parsePredictions(const Matrix&) const = 0;
        virtual Matrix predict(const Matrix&) const;
        const string& getProgressMetricName() const;
        virtual void setFeatureScalar(Scalar*);
        virtual void setTargetScalar(Scalar*);
        virtual void resetToRaw();
        virtual void fitScalars(Matrix&, vector<double>&, Matrix&, vector<double>&);
        virtual double processBatch(Batch&, vector<double>&, const Matrix&, const Loss*) const = 0;
        virtual double calculateProgressMetric(const Batch&, const Matrix&, const vector<double>&, EpochStats&) const = 0;
        virtual ~Task();
};