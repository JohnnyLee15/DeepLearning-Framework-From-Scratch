#include "core/model/Pipeline.h"
#include "utils/ConsoleUtils.h"
#include "utils/BinUtils.h"
#include "core/data/Data.h"
#include <cstdint>
#include "utils/Scalar.h"
#include "core/data/TabularData.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"
#include "utils/ImageTransform2D.h"
#include "core/data/ImageData2D.h"

Pipeline::Pipeline() : 
    data(nullptr), 
    featureScalar(nullptr), 
    targetScalar(nullptr), 
    model(nullptr),
    imageTransformer(nullptr),
    isLoadedPipeline(false) {}

Pipeline::~Pipeline() {
    delete model;
    delete data;
    delete featureScalar;
    delete targetScalar;
    delete imageTransformer;
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
    if (data != nullptr) {
        delete data;
    }
    data = newData;
}

void Pipeline::setFeatureScalar(Scalar *newFeatureScalar) {
    checkIsLoadedPipeline("Replacing the feature scalar");
    if (featureScalar != nullptr) {
        delete featureScalar;
    }
    featureScalar = newFeatureScalar;
}

void Pipeline::setTargetScalar(Scalar *newTargetScalar) {
    checkIsLoadedPipeline("Replacing the target scalar");
    if (targetScalar != nullptr) {
        delete targetScalar;
    }
    targetScalar = newTargetScalar;
}

void Pipeline::setModel(NeuralNet *nn) {
    checkIsLoadedPipeline("Replacing the neural network");\
    if (model != nullptr) {
        delete model;
    }
    model = nn;
}

void Pipeline::setImageTransformer2D(ImageTransform2D *transformer) {
    checkIsLoadedPipeline("Replacing the image transformer");
    if (imageTransformer != nullptr) {
        delete imageTransformer;
    }
    imageTransformer = transformer;
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

ImageTransform2D* Pipeline::getImageTransformer() const {
    return imageTransformer;
}

void Pipeline::saveToBin(const string &filename) const {
    BinUtils::savePipeline(*this, filename);
}

void Pipeline::writeBin(ofstream &modelBin) const {
    uint8_t hasModel = (model == nullptr) ? 0 : 1;
    modelBin.write((char*) &hasModel, sizeof(uint8_t));
    if (model) {
        model->writeBin(modelBin);
    }

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

    uint8_t hasTransformer = (imageTransformer == nullptr) ? 0 : 1;
    modelBin.write((char*) &hasTransformer, sizeof(uint8_t));
    if (imageTransformer) {
        imageTransformer->writeBin(modelBin);
    }
}

Pipeline Pipeline::loadFromBin(const string &filename) {
    return BinUtils::loadPipeline(filename);
}

void Pipeline::loadModel(ifstream &modelBin) {
    uint8_t hasModel;
    modelBin.read((char*) &hasModel, sizeof(uint8_t));
    
    if (hasModel) {
        model = new NeuralNet();
        model->loadFromBin(modelBin);
    }
}

void Pipeline::loadData(ifstream &modelBin) {
    uint32_t dataEncoding;
    modelBin.read((char*) &dataEncoding, sizeof(uint32_t));

    if (dataEncoding == Data::Encodings::Tabular) {
        data = new TabularData();
    } else if (dataEncoding == Data::Encodings::Image2D) {
        data = new ImageData2D();
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

void Pipeline::loadImageTransformer2D(ifstream &modelBin) {
    uint8_t hasTransformer;
    modelBin.read((char*) &hasTransformer, sizeof(uint8_t));

    if (hasTransformer) {
        imageTransformer = new ImageTransform2D();
        imageTransformer->loadFromBin(modelBin);
    }
}

void Pipeline::loadComponents(ifstream &modelBin) {
    loadModel(modelBin);
    loadData(modelBin);
    loadFeatureScalar(modelBin);
    loadTargetScalar(modelBin);
    loadImageTransformer2D(modelBin);
    isLoadedPipeline = true;
}