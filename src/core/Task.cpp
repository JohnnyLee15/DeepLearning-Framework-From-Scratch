#include "core/Task.h"
#include <iostream>
#include "core/Matrix.h"
#include "utils/Scalar.h"
#include "utils/ConsoleUtils.h"

Task::Task(string progressMetricName) : 
    progressMetricName(progressMetricName), featureScalar(nullptr) {}

Matrix Task::predict(const Matrix &activations) const {
    return activations;
}

void Task::setTargetScalar(Scalar *scalar) {
    ConsoleUtils::fatalError("This task does not support target scaling.");
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
        ConsoleUtils::fatalError("Feature scalar not set before resetToRaw().");
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
        ConsoleUtils::fatalError("Feature scalar must be set before calling fitScalars().");
    }

    featureScalar->fit(trainFeatures);
    featureScalar->transform(trainFeatures);
    featureScalar->transform(testFeatures);
}

Task::~Task() {
    delete featureScalar;
}
