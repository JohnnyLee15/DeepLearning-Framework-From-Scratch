#include "core/Pipeline.h"
#include "utils/ConsoleUtils.h"
#include "utils/BinUtils.h"
#include "core/Data.h"
#include <cstdint>
#include "utils/Scalar.h"
#include "core/TabularData.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"

Pipeline::Pipeline() : 
    data(nullptr), featureScalar(nullptr), targetScalar(nullptr), isLoadedPipeline(false) {}

Pipeline::~Pipeline() {
    delete model;
    delete data;
    delete featureScalar;
    delete targetScalar;
}

void Pipeline::checkIsLoadedPipeline(const string &message) const {
    if (isLoadedPipeline) {
        ConsoleUtils::fatalError(
            "This pipeline was loaded from a saved model. "
            + message + " is not allowed to ensure consistency."
        );
    }
}

void Pipeline::setData(Data *newData) {
    checkIsLoadedPipeline("Replacing the data object");
    data = newData;
}

void Pipeline::setFeatureScalar(Scalar *newFeatureScalar) {
    checkIsLoadedPipeline("Replacing the feature scalar");
    featureScalar = newFeatureScalar;
}

void Pipeline::setTargetScalar(Scalar *newTargetScalar) {
    checkIsLoadedPipeline("Replacing the target scalar");
    targetScalar = newTargetScalar;
}

void Pipeline::setModel(NeuralNet *nn) {
    checkIsLoadedPipeline("Replacing the neural network");
    model = nn;
}

Data* Pipeline::getData() const {
    return data;
}

Scalar* Pipeline::getFeatureScalar() const {
    return featureScalar;
}

Scalar* Pipeline::getTargetScalar() const {
    return targetScalar;
}

NeuralNet* Pipeline::getModel() const {
    return model;
}

void Pipeline::saveToBin(const string &filename) const {
    BinUtils::savePipeline(*this, filename);
}

void Pipeline::writeBin(ofstream &modelBin) const {
    model->writeBin(modelBin);

    if (data) {
        data->writeBin(modelBin);
    } else {
        uint32_t dataEncoding = Data::Encodings::None;
        modelBin.write((char*) &dataEncoding, sizeof(uint32_t));
    }

    if (featureScalar) {
        featureScalar->writeBin(modelBin);
    } else {
        uint32_t featureScalarEncoding = Scalar::Encodings::None;
        modelBin.write((char*) &featureScalarEncoding, sizeof(uint32_t));
    }


    if (targetScalar) {
        targetScalar->writeBin(modelBin);
    } else {
        uint32_t targetScalarEncoding = Scalar::Encodings::None;
        modelBin.write((char*) &targetScalarEncoding, sizeof(uint32_t));
    }
}

Pipeline Pipeline::loadFromBin(const string &filename) {
    return BinUtils::loadPipeline(filename);
}

void Pipeline::loadData(ifstream &modelBin) {
    uint32_t dataEncoding;
    modelBin.read((char*) &dataEncoding, sizeof(uint32_t));

    if (dataEncoding == Data::Encodings::Tabular) {
        data = new TabularData();
    } else if (dataEncoding == Data::Encodings::Image2D) {
        // TO DO
    } else if (dataEncoding != Data::Encodings::None){
        ConsoleUtils::fatalError(
            "Unsupported scalar encoding \"" + to_string(dataEncoding) + "\"."
        ); 
    }

    if (data) {
        data->loadFromBin(modelBin);
    }
}

void Pipeline::loadFeatureScalar(ifstream &modelBin) {
    uint32_t scalarEncoding;
    modelBin.read((char*) &scalarEncoding, sizeof(uint32_t));

    if (scalarEncoding == Scalar::Encodings::Greyscale) {
        featureScalar = new Greyscale();
    } else if(scalarEncoding == Scalar::Encodings::Minmax)  {
        featureScalar = new Minmax();
    } else if (scalarEncoding != Scalar::Encodings::None) {
        ConsoleUtils::fatalError(
            "Unsupported scalar encoding \"" + to_string(scalarEncoding) + "\"."
        ); 
    }

    if (featureScalar) {
        featureScalar->loadFromBin(modelBin);
    }
}

void Pipeline::loadTargetScalar(ifstream &modelBin) {
    uint32_t scalarEncoding;
    modelBin.read((char*) &scalarEncoding, sizeof(uint32_t));

    if (scalarEncoding == Scalar::Encodings::Greyscale) {
        targetScalar = new Greyscale();
    } else if(scalarEncoding == Scalar::Encodings::Minmax)  {
        targetScalar = new Minmax();
    } else if (scalarEncoding != Scalar::Encodings::None) {
        ConsoleUtils::fatalError(
            "Unsupported scalar encoding \"" + to_string(scalarEncoding) + "\"."
        ); 
    }

    if (targetScalar) {
        targetScalar->loadFromBin(modelBin);
    }
}

void Pipeline::loadComponents(ifstream &modelBin) {
    model = new NeuralNet();
    model->loadFromBin(modelBin);
    loadData(modelBin);
    loadFeatureScalar(modelBin);
    loadTargetScalar(modelBin);
    isLoadedPipeline = true;
}