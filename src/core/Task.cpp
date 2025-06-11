#include "core/Task.h"
#include <iostream>
#include "core/Matrix.h"
#include "utils/Scalar.h"

Task::Task(string progressMetricName) : 
    progressMetricName(progressMetricName), featureScalar(nullptr) {}

Matrix Task::predict(const Matrix &activations) const {
    return activations;
}

void Task::setTargetScalar(Scalar *scalar) {
    cerr << "Error: This task does not support target scaling." << endl;
    exit(1);
}

const string& Task::getProgressMetricName() const {
    return progressMetricName;
}

void Task::setFeatureScalar(Scalar *scalar) {
    if (featureScalar != nullptr) {
        delete featureScalar;
    }
    featureScalar = scalar;
}

void Task::resetToRaw() {
    if (!featureScalar) {
        std::cerr << "Fatal Error: Feature scalar not set before resetToRaw()." << std::endl;
        exit(1);
    }

    featureScalar->resetToRaw();
}

void Task::fitScalars(
    Matrix &trainFeatures,
    vector<double> &trainTargets,
    Matrix &testFeatures,
    vector<double> &testTargets
) {
    if (!featureScalar) {
        std::cerr << "Fatal Error: Feature scalar must be set before calling fitScalars()." << std::endl;
        exit(1);
    }

    featureScalar->fit(trainFeatures);
    featureScalar->transform(trainFeatures);
    featureScalar->transform(testFeatures);
}

Task::~Task() {
    delete featureScalar;
}
