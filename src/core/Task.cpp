#include "core/Task.h"
#include <iostream>
#include "utils/Scalar.h"
#include "utils/ConsoleUtils.h"
#include "core/Tensor.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"

Task::Task(string progressMetricName) : 
    progressMetricName(progressMetricName), featureScalar(nullptr) {}

Tensor Task::predict(const Tensor &activations) const {
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
    Tensor &features,
    vector<double> &targets
) {
    if (!featureScalar) {
        ConsoleUtils::fatalError("Feature scalar must be set before calling fitScalars().");
    }

    featureScalar->fit(features);
}

void Task::transformScalars(
    Tensor &features,
    vector<double> &targets
) {
    if (!featureScalar) {
        ConsoleUtils::fatalError(
            "Feature scalar must be set and fit before calling transformScalars()."
        );
    }

    featureScalar->transform(features);
}

void Task::reverseTransformScalars(
    Tensor &features,
    vector<double> &targets
) {
    if (!featureScalar) {
        ConsoleUtils::fatalError(
            "Feature scalar must be set and fit before calling reverseTransformScalars()."
        );
    }

    featureScalar->reverseTransform(features);
}

void Task::writeBin(ofstream &modelBin) const {
    uint32_t taskEncoding = getEncoding();
    modelBin.write((char*) &taskEncoding, sizeof(uint32_t));

    if (featureScalar) {
        featureScalar->writeBin(modelBin);
    } else {
        uint32_t featureScalarEncodng = Scalar::Encodings::None;
        modelBin.write((char*) &featureScalarEncodng, sizeof(uint32_t));
    }
}

void Task::loadFromBin(ifstream &modelBin) {
    uint32_t scalarEncoding;
    modelBin.read((char*) &scalarEncoding, sizeof(uint32_t));

    if (scalarEncoding == Scalar::Encodings::Greyscale) {
        featureScalar = new Greyscale();
    } else if(scalarEncoding == Scalar::Encodings::Minmax)  {
        featureScalar = new Minmax();
    } else if (scalarEncoding == Scalar::Encodings::None) {
        featureScalar = nullptr;
    } else {
        ConsoleUtils::fatalError(
            "Unsupported scalar encoding \"" + to_string(scalarEncoding) + "\"."
        ); 
    }

    if (featureScalar) {
        featureScalar->loadFromBin(modelBin);
    }
}

Task::~Task() {
    delete featureScalar;
}
