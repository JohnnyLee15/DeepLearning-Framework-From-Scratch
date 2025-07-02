#pragma once

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include "core/Tensor.h"
#include "core/Task.h"
#include "core/Data.h"

using namespace std;

class TabularData : public Data {
    private:
        // Constants
        static const size_t NO_TARGET_IDX;
        static const string NO_TARGET_COL;
        static const size_t MAX_DISPLAY_COLS;

        // Instances Variables
        vector<string> header;
        Tensor rawTrainFeatures;
        vector<double> rawTrainTargets;
        Tensor rawTestFeatures;
        vector<double> rawTestTargets;
        Tensor trainFeatures;
        vector<double> trainTargets;
        Tensor testFeatures;
        vector<double> testTargets;
        Task *task;
        bool isDataLoaded;
        vector<bool> isCategorical;
        vector<unordered_map<string, double> > featureEncodings;
        bool isLoadedFromModel;

        // Static Variables
        static random_device rd;
        static mt19937 generator;

        // Methods
        void readCsv(string, bool, size_t, const string&, bool);
        void setData(const Tensor&, vector<double>&, bool);
        void head(size_t, const Tensor&) const;
        void checkDataLoaded() const;
        void checkTask(const string&) const;
        void parseRawData(vector<vector<string> >&, vector<string>&, const vector<string>&, size_t);
        size_t getColIdx(const string&) const;
        vector<string> validateAndLoadCsv(const string&, bool);
        Tensor readFeatures(const vector<vector<string> >&);
        
    public:
        // Constructor
        TabularData();

        // Methods
        void readTrain(string, size_t, bool header = false);
        void readTest(string, size_t, bool header = false);
        void readTrain(string, const string&);
        void readTest(string, const string&);
        // void readAllData(string, int, float);
        const Tensor& getTrainFeatures() const override;
        const Tensor& getTestFeatures() const override;
        const vector<double>& getTrainTargets() const override;
        const vector<double>& getTestTargets() const override;
        size_t getNumTrainSamples() const override;
        const Task* getTask() const override;
        vector<size_t> generateShuffledIndices() const override;
        void headTrain(size_t numRows = 6) const;
        void headTest(size_t numRows = 6) const;
        void setTask(Task*);
        void resetToRaw();
        void fitScalars();
        void setScalars(Scalar*, Scalar *targetScalar = nullptr);
        void transformTrain();
        void transformTest();
        void reverseTransformTrain();
        void reverseTransformTest();
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;

        ~TabularData();
};