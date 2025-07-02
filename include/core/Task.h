#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "utils/EpochStats.h"
#include <cstdint>

class Batch;
class DenseLayer;
class Tensor;
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
        virtual Tensor predict(const Tensor&) const;
        const string& getProgressMetricName() const;
        virtual void setFeatureScalar(Scalar*);
        virtual void setTargetScalar(Scalar*);
        virtual void resetToRaw();
        virtual void fitScalars(Tensor&, vector<double>&);
        virtual void transformScalars(Tensor&, vector<double>&);
        virtual void reverseTransformScalars(Tensor&, vector<double>&);
        virtual double processBatch(Batch&, vector<double>&, const Tensor&, const Loss*) const = 0;
        virtual double calculateProgressMetric(const Batch&, const Tensor&, const vector<double>&, EpochStats&) const = 0;
        virtual void writeBin(ofstream&) const = 0;
        virtual void loadFromBin(ifstream&);
        virtual ~Task();
        virtual uint32_t getEncoding() const = 0;

        // Enums
        enum Encodings : uint32_t {
            Classification,
            Regression
        };
};