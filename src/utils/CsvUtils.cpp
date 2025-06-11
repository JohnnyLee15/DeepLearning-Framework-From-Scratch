#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <cctype>

void CsvUtils::checkFile(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Fatal Error: Unable to open file \"" << filename << "\" for reading.\n"
             << "Please check that the file exists and the path is correct." << endl;
        exit(1);
    }
}

vector<string> CsvUtils::collectLines(const string &filename, bool hasHeader) {
    ConsoleUtils::loadMessage("Loading Data.");
    ifstream file(filename);
    vector<string> lines;
    string line;

    if (hasHeader) {
        getline(file, line);
    }

    while(getline(file, line)) {
        lines.push_back(line);
    }

    ConsoleUtils::completeMessage();

    if (lines.empty()) {
        cerr << "Fatal Error: No samples found in file \"" << filename << "\".\n"
            << "The file may be empty or improperly formatted." << endl;
        exit(1);
    }

    return lines;
}

void CsvUtils::parseLines(
    const vector<string> &lines,
    vector<vector<string> > &featuresRaw,
    vector<string> &targetsRaw,
    size_t targetIdx
) {
    ConsoleUtils::loadMessage("Parsing Lines.");
    size_t numSamples = lines.size();

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        parseLine(lines[i], featuresRaw[i], targetsRaw[i], targetIdx);
    }
    ConsoleUtils::completeMessage();
}

void CsvUtils::validateField(const string &token, const string &line) {
    if (token.empty()) {
        cerr << "Fatal Error: Missing field detected in line.\n"
            << "Line: \"" << line << "\"" << endl;
        exit(1);
    }
}

void CsvUtils::parseLine(
    const string &line, 
    vector<string> &sample, 
    string &target, 
    size_t targetIdx
) {
    size_t numCols = sample.size() + 1;
    stringstream lineParser(line);
    string token;

    size_t currIdx = 0;
    size_t featureIdx = 0;
    while(getline(lineParser, token, ',') && currIdx < numCols) {
        token = toLowerCase(trim(token));
        validateField(token, line);

        if (currIdx == targetIdx) {
            target = token;
        } else {
            sample[featureIdx++] = token;
        }

        currIdx++;
    }

    if (currIdx != numCols) {
        cerr << "Fatal Error: Row does not match expected column count.\n"
            << "Expected " << numCols << " fields, but got " << currIdx << " (or more).\n"
            << "Line: \"" << line << "\"" << endl;
        exit(1);
    }
}

vector<string> CsvUtils::readHeader(const string &filename) {
    ifstream file(filename);
    string headerLine;

    if (!getline(file, headerLine)) {
        cerr << "Fatal Error: Failed to read header from file \"" << filename << "\".\n"
             << "The file may be empty or incorrectly formatted." << endl;
        exit(1);
    }

    stringstream lineParser(headerLine);
    string token;
    vector<string> header;
    while(getline(lineParser, token, ',')) {
        header.push_back(toLowerCase(trim(token)));
    }
    
    return header;
}

string CsvUtils::trim(const string &str) {
    size_t length = str.size();

    size_t start = 0;
    while(start < length && isspace(str[start])) {
        start++;
    }

    size_t end = length;
    while (end > start && isspace(str[end - 1])) {
        end--;
    }

    return str.substr(start, end - start);
}

string CsvUtils::toLowerCase(const string &str) {
    size_t length = str.length();
    string strLower;
    strLower.reserve(length);
    char shiftSize = 'a' - 'A';

    for (size_t i = 0; i < length; i++) {
        if (str[i] <= 'Z' && str[i] >= 'A') {
            strLower += (str[i] + shiftSize);
        } else {
            strLower += str[i];
        }
    }

    return strLower;
}

size_t CsvUtils::countFirstCol(const string &firstLine) {
    stringstream lineParser(firstLine);
    string token;

    size_t numCols = 0;
    while(getline(lineParser, token, ',')) {
        numCols++;
    }
    
    return numCols;
}
