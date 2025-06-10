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
    cout << "Error: This task does not support target scaling." << endl;
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
    featureScalar->resetToRaw();
}

void Task::fitScalars(
    Matrix &trainFeatures,
    vector<double> &trainTargets,
    Matrix &testFeatures,
    vector<double> &testTargets
) {
    featureScalar->fit(trainFeatures);
    featureScalar->transform(trainFeatures);
    featureScalar->transform(testFeatures);
}

Task::~Task() {
    delete featureScalar;
}
