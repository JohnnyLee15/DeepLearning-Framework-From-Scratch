#include "utils/BinUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include <cerrno>
#include <cstring>
#include "core/model/Pipeline.h"
#include "core/model/NeuralNet.h"
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

const char BinUtils::CANCEL = 'q';
const char BinUtils::OVERRIDE = 'o';
const char BinUtils::RENAME = 'r';
const string BinUtils::MODEL_EXTENSION = ".nn";

void BinUtils::savePipeline(const Pipeline &pipe, const string &filepath) {
    string fileToWrite = addExtension(filepath);
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
            fileToWrite = addExtension(getNewModelPath());
            done = !fileExists(fileToWrite, true);
            if (done) {
                shouldWrite = true;
            }
        }
    }

    if (shouldWrite) {
        checkParentDirs(fileToWrite);
        writeToBin(pipe, fileToWrite);
        ConsoleUtils::printSuccess("Model saved successfully as \"" + fileToWrite + "\".", true);
    }

    ConsoleUtils::printSepLine();
}

bool BinUtils::fileExists(string filepath, bool showLineSep) {
    ifstream file(filepath);
    if (file.good()) {
        if (showLineSep) {
            ConsoleUtils::printSepLine();
        }
        cout << endl;
        ConsoleUtils::printWarning("File \"" + filepath + "\"" + " already exists.");
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

void BinUtils::checkParentDirs(const string &filepath) {
    fs::path finalPath = filepath;
    fs::path parent = finalPath.parent_path();
    if (!parent.empty()) {
        error_code ec;
        fs::create_directories(parent, ec);

        if (ec && (!fs::exists(parent) || !fs::is_directory(parent))) {
            ConsoleUtils::fatalError(
                "Cannot create parent directory \"" + parent.string() +
                "\" for \"" + filepath + "\": " + ec.message() +
                ". Choose a different save path or create the folder and retry."
            );
        }
    }
}

void BinUtils::writeToBin(const Pipeline &pipe, const string &filepath) {
    ofstream modelBin(filepath, ios::out | ios::binary);
    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Could not open \"" + filepath + "\": " + strerror(errno) + "."
        );
    }

    pipe.writeBin(modelBin);

    modelBin.close();
    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Failed to write model to \"" + filepath + "\". The file may be corrupted."
        );
    }
}

string BinUtils::getNewModelPath() {
    bool done = false;
    string newFilepath;
    while (!done) {
        cout << "[>] Enter the new model path: ";
        getline(cin, newFilepath);
        newFilepath = CsvUtils::trim(newFilepath);

        if (newFilepath.length() > 0) {
            done = true;
        } else {
            ConsoleUtils::printError("Error: File name must contain atleast one character.");
        }
    }

    return newFilepath;
}


Pipeline BinUtils::loadPipeline(const string &filepath) {
    string fullFilepath = addExtension(filepath);

    ifstream modelBin(fullFilepath, ios::in | ios::binary);
    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Unable to open file \"" + fullFilepath + "\" for reading.\n" +
            "Reason: " + strerror(errno) + "."
        );
    }

    Pipeline pipe;
    pipe.loadComponents(modelBin);
    modelBin.close();

    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Failed to fully read model from \"" + fullFilepath + "\". "
            "The file may be corrupted or incomplete."
        );
    }

    ConsoleUtils::printSuccess("Model successfully loaded from \"" + fullFilepath + "\".", true);
    ConsoleUtils::printSepLine();
    return pipe;
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

void BinUtils::saveBestWeights(const string &path, const NeuralNet &nn) {
    ofstream modelBin(path, ios::out | ios::binary);
    if (!modelBin) {
        ConsoleUtils::printError(
            "Could not open best weights from \"" + path + "\": " + strerror(errno) + "."
        );
        return;
    }

    nn.saveBestWeights(modelBin);

    if (!modelBin) {
        ConsoleUtils::printError(
            "Failed to write best weights to \"" + path + "\"."
        );
    }
}

void BinUtils::loadBestWeights(const string &path, NeuralNet &nn) {
    ifstream modelBin(path, ios::in | ios::binary);
    if (!modelBin) {
        ConsoleUtils::printError(
            "Could not open best weights from \"" + path + "\": " + strerror(errno) + "."
        );
        return;
    }

    nn.loadBestWeights(modelBin);
    modelBin.close();

    if (!modelBin) {
        ConsoleUtils::printError(
            "Failed to fully read best weights from \"" + path + "\". "
        );
        return;
    }
}