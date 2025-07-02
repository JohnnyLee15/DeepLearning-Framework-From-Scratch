#include "utils/BinUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "core/Layer.h"
#include "core/DenseLayer.h"
#include "core/NeuralNet.h"
#include "losses/Loss.h"
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include <cerrno>
#include <cstring>
#include "losses/MSE.h"
#include "losses/SoftmaxCrossEntropy.h"
#include "activations/Activation.h"
#include "activations/Linear.h"
#include "activations/ReLU.h"
#include "activations/Softmax.h"
#include "core/Data.h"

const char BinUtils::CANCEL = 'q';
const char BinUtils::OVERRIDE = 'o';
const char BinUtils::RENAME = 'r';
const string BinUtils::MODEL_EXTENSION = ".nn";

void BinUtils::saveModel(
    const NeuralNet &nn, 
    const string &filename, 
    const Data &data
) {
    string fileToWrite = addExtension(filename);
    bool done = !fileExists(fileToWrite, false);
    bool shouldWrite = done;
    while (!done) {
        char choice = getUserChoice();
        if (choice == CANCEL) {
            ConsoleUtils::printError("Model save cancelled.");
            done = true;
        } else if (choice == OVERRIDE) {
            done = true;
            shouldWrite = true; 
            ConsoleUtils::printWarning("Overwriting existing model.");
        } else if (choice == RENAME) {
            fileToWrite = addExtension(getNewModelName());
            done = !fileExists(fileToWrite, true);
            if (done) {
                shouldWrite = true;
            }
        }
    }

    if (shouldWrite) {
        writeToBin(nn, fileToWrite, data);
        ConsoleUtils::printSuccess("Model saved successfully as \"" + fileToWrite + "\".");
    }

    ConsoleUtils::printSepLine();
}

bool BinUtils::fileExists(string filename, bool showLineSep) {
    ifstream file(filename);
    if (file.good()) {
        if (showLineSep) {
            ConsoleUtils::printSepLine();
        }
        cout << endl;
        ConsoleUtils::printWarning("File \"" + filename + "\"" + " already exists.");
    }
    return file.good();
}

void BinUtils::printOptions() {
    cout << "   [q] Cancel save." << endl;
    cout << "   [o] Overwrite existing file." << endl;
    cout << "   [r] Rename and save as new file." << endl;
}

char BinUtils::getUserChoice() {
    bool done = false;
    char choice;
    printOptions();
    while (!done) {
        cout << "\n[>] Enter your choice: ";
        string input;
        getline(cin, input);
        input = CsvUtils::toLowerCase(CsvUtils::trim(input));

        if (input.length() != 1) {
            ConsoleUtils::printError("Invalid input. Please enter a single character: [q], [o], or [r].");
        } else {
            choice = input[0];
            if (choice == CANCEL || choice == OVERRIDE || choice == RENAME) {
                done = true;
            } else {
                ConsoleUtils::printError("Invalid input. Please enter one of: [q], [o], [r].");
            }
        }
    }

    return choice;
}

void BinUtils::writeToBin(
    const NeuralNet &nn, 
    const string &filename,
    const Data &data
) {
    ofstream modelBin(filename, ios::out | ios::binary);
    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Could not open \"" + filename + "\": " + strerror(errno) + "."
        );
    } 

    const Loss *loss = nn.getLoss();
    const vector<Layer*> &layers = nn.getLayers();

    uint32_t lossEncoding = loss->getEncoding();
    modelBin.write((char*) &lossEncoding, sizeof(uint32_t));

    uint32_t numActiveLayers = layers.size();
    modelBin.write((char*) &numActiveLayers, sizeof(uint32_t));

    for (uint32_t i = 0; i < numActiveLayers; i++) {
        layers[i]->writeBin(modelBin);
    }

    data.writeBin(modelBin);
    modelBin.close();

    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Failed to write model to \"" + filename + "\". The file may be corrupted."
        );
    }
}

string BinUtils::getNewModelName() {
    bool done = false;
    string newFilename;
    while (!done) {
        cout << "[>] Enter the new model name: ";
        getline(cin, newFilename);
        newFilename = CsvUtils::trim(newFilename);

        if (newFilename.length() > 0) {
            done = true;
        } else {
            ConsoleUtils::printError("Error: File name must contain atleast one character.");
        }
    }

    return newFilename;
}

Loss* BinUtils::loadLoss(ifstream &modelBin) {
    uint32_t lossEncoding;
    modelBin.read((char*) &lossEncoding, sizeof(uint32_t));

    Loss *loss = nullptr;
    if (lossEncoding == Loss::Encodings::MSE) {
        loss = new MSE();
    } else if (lossEncoding == Loss::Encodings::SoftmaxCrossEntropy) {
        loss = new SoftmaxCrossEntropy();
    } else {
        ConsoleUtils::fatalError(
            "Unsupported loss encoding \"" + to_string(lossEncoding) + "\" in model file."
        );
    } 

    return loss;
}

Activation* BinUtils::loadActivation(ifstream &modelBin) {
    uint32_t activationEncoding;
    modelBin.read((char*) &activationEncoding, sizeof(uint32_t));

    Activation *activation = nullptr;
    if (activationEncoding == Activation::Encodings::Linear){
        activation = new Linear();
    } else if (activationEncoding == Activation::Encodings::ReLU) {
        activation = new ReLU();
    } else if (activationEncoding == Activation::Encodings::Softmax) {
        activation = new Softmax();
    } else {
        ConsoleUtils::fatalError(
            "Unsupported activation encoding \"" + to_string(activationEncoding) + "\"."
        );
    }

    return activation;
}

Layer* BinUtils::loadLayer(ifstream &modelBin) {
    uint32_t layerEncoding;
    modelBin.read((char*) &layerEncoding, sizeof(uint32_t));

    Activation *activation = loadActivation(modelBin);

    Layer *layer = nullptr;
    if (layerEncoding == Layer::Encodings::DenseLayer) {
        uint32_t numNeurons;
        modelBin.read((char*) &numNeurons, sizeof(uint32_t));

        uint32_t numWeights;
        modelBin.read((char*) &numWeights, sizeof(uint32_t));

        layer = new DenseLayer(numNeurons, numWeights, activation);
    } else{
        ConsoleUtils::fatalError(
            "Unsupported layer encoding \"" + to_string(layerEncoding) + "\"."
        );
    }

    layer->loadWeightsAndBiases(modelBin);
    return layer;
}

NeuralNet BinUtils::loadModel(const string &filename, Data &data) {
    string fullFilename = addExtension(filename);

    ifstream modelBin(fullFilename, ios::in | ios::binary);
    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Unable to open file \"" + fullFilename + "\" for reading.\n" +
            "Reason: " + strerror(errno) + "."
        );
    }

    Loss *loss = loadLoss(modelBin);
    uint32_t numActiveLayers;
    modelBin.read((char*) &numActiveLayers, sizeof(uint32_t));

    vector<Layer*> layers(numActiveLayers);
    for (uint32_t i = 0; i < numActiveLayers; i++) {
        layers[i] = loadLayer(modelBin);
    }

    data.loadFromBin(modelBin);
    modelBin.close();

    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Failed to fully read model from \"" + fullFilename + "\". "
            "The file may be corrupted or incomplete."
        );
    }

    ConsoleUtils::printSuccess("Model successfully loaded from \"" + fullFilename + "\".");
    ConsoleUtils::printSepLine();
    return NeuralNet(layers, loss);
}

string BinUtils::addExtension(const string &modelName) {
    size_t extLength = MODEL_EXTENSION.length();
    size_t nameLength = modelName.length();

    if (nameLength >= extLength) {
        bool isMatching = true;
        for (size_t i = 0; i < extLength && isMatching; i++) {
            if (modelName[i + nameLength - extLength] != MODEL_EXTENSION[i]) {
                isMatching = false;
            }
        }

        if (!isMatching) {
            return modelName + MODEL_EXTENSION;
        }
    }

    return modelName;
}