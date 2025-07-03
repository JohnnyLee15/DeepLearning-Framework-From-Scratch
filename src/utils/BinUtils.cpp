#include "utils/BinUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include <cerrno>
#include <cstring>
#include "core/Pipeline.h"

const char BinUtils::CANCEL = 'q';
const char BinUtils::OVERRIDE = 'o';
const char BinUtils::RENAME = 'r';
const string BinUtils::MODEL_EXTENSION = ".nn";

void BinUtils::savePipeline(const Pipeline &pipe, const string &filename) {
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
        writeToBin(pipe, fileToWrite);
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

void BinUtils::writeToBin(const Pipeline &pipe, const string &filename) {
    ofstream modelBin(filename, ios::out | ios::binary);
    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Could not open \"" + filename + "\": " + strerror(errno) + "."
        );
    }

    pipe.writeBin(modelBin);

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


Pipeline BinUtils::loadPipeline(const string &filename) {
    string fullFilename = addExtension(filename);

    ifstream modelBin(fullFilename, ios::in | ios::binary);
    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Unable to open file \"" + fullFilename + "\" for reading.\n" +
            "Reason: " + strerror(errno) + "."
        );
    }

    Pipeline pipe;
    pipe.loadComponents(modelBin);
    modelBin.close();

    if (!modelBin) {
        ConsoleUtils::fatalError(
            "Failed to fully read model from \"" + fullFilename + "\". "
            "The file may be corrupted or incomplete."
        );
    }

    ConsoleUtils::printSuccess("Model successfully loaded from \"" + fullFilename + "\".");
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