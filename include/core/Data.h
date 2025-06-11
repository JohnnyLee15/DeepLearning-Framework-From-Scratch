#pragma once

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include "core/Matrix.h"
#include "core/Task.h"

using namespace std;

class Data {
    private:
        // Constants
        static const size_t NO_TARGET_IDX;
        static const string NO_TARGET_COL;
        static const size_t MAX_DISPLAY_COLS;

        // Instances Variables
        vector<string> header;
        Matrix rawTrainFeatures;
        vector<double> rawTrainTargets;
        Matrix rawTestFeatures;
        vector<double> rawTestTargets;
        Matrix trainFeatures;
        vector<double> trainTargets;
        Matrix testFeatures;
        vector<double> testTargets;
        Task *task;
        bool isDataLoaded;

        // Static Variables
        static random_device rd;
        static mt19937 generator;

        // Methods
        void readCsv(string, bool, size_t, const string&, bool);
        void setData(const Matrix&, vector<double>&, bool);
        void head(size_t, const Matrix&) const;
        void checkDataLoaded() const;
        void parseRawData(vector<vector<string> >&, vector<string>&, const vector<string>&, size_t);
        size_t getColIdx(const string&) const;
        vector<string> validateAndLoadCsv(const string&, bool);
        
    public:
        // Constructor
        Data();

        // Methods
        void readTrain(string, size_t, bool header = false);
        void readTest(string, size_t, bool header = false);
        void readTrain(string, const string&);
        void readTest(string, const string&);
        // void readAllData(string, int, float);
        const Matrix& getTrainFeatures() const;
        const Matrix& getTestFeatures() const;
        const vector<double>& getTrainTargets() const;
        const vector<double>& getTestTargets() const;
        size_t getNumTrainSamples() const;
        const Task* getTask() const;
        vector<size_t> generateShuffledIndices() const;
        void headTrain(size_t numRows = 6) const;
        void headTest(size_t numRows = 6) const;
        void setTask(Task*);
        void resetToRaw();
        void fitScalars();
        void setScalars(Scalar*, Scalar *targetScalar = nullptr);

        ~Data();
};