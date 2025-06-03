#pragma once

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include "core/Matrix.h"

using namespace std;

class Data {
    private:
        // Instances Variables
        unordered_map<string, int> labelMap;
        vector<string> header;

        Matrix trainFeatures;
        vector<int> trainTarget;

        Matrix testFeatures;
        vector<int> testTarget;

        bool isDataLoaded;

        // Constants
        static const int NO_TARGET_IDX;
        static const string NO_TARGET_COL;
        static const size_t MAX_DISPLAY_COLS;

        // Static Variables
        static random_device rd;
        static mt19937 generator;

        // Private Methods
        void readCsv(string, bool, int, const string&, bool);
        vector<int> getTarget(const vector<string>&);
        void createLabelMap(const vector<string>&);
        void setData(const Matrix&, vector<int>&, bool);
        int getColIdx(const string&) const;
        void head(size_t, const Matrix&) const;
    
    public:
        Data();
        void readTrain(string, int, bool header = false);
        void readTest(string, int, bool header = false);
        void readTrain(string, const string&);
        void readTest(string, const string&);
        // void readAllData(string, int, float);
        const Matrix& getTrainFeatures() const;
        const Matrix& getTestFeatures() const;
        const vector<int>& getTrainTarget() const;
        const vector<int>& getTestTarget() const;
        void minmax();
        void minmaxGreyScale();
        size_t getTrainFeatureSize() const;
        vector<int> generateShuffledIndices() const;
        void headTrain(size_t numRows = 6) const;
        void headTest(size_t numRows = 6) const;
};