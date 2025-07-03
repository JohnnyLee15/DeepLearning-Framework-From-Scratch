#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "core/Tensor.h"
#include "core/Data.h"

using namespace std;

class TabularData : public Data {
    private:
        // Constants
        static const size_t NO_TARGET_IDX;
        static const string NO_TARGET_COL;
        static const string REGRESSION_TASK;
        static const string CLASSIFICATION_TASK;
        static const size_t MAX_DISPLAY_COLS;

        // Instances Variables
        vector<string> header;

        Tensor trainFeatures;
        Tensor testFeatures;
        vector<double> trainTargets;
        vector<double> testTargets;

        vector<bool> isCategorical;
        vector<unordered_map<string, double> > featureEncodings;

        unordered_map<string, int> labelMap;

        bool isTrainLoaded;
        bool isTestLoaded;
        bool isLoadedFromModel;

        string task;

        // Methods
        void readCsv(string, bool, size_t, const string&, bool);
        void setData(const Tensor&, vector<double>&, bool);
        void head(size_t, const Tensor&) const;
        void checkTrainLoaded() const;
        void checkTestLoaded() const;
        void parseRawData(vector<vector<string> >&, vector<string>&, const vector<string>&, size_t);
        size_t getColIdx(const string&) const;
        vector<string> validateAndLoadCsv(const string&, bool);
        Tensor readFeatures(const vector<vector<string> >&);
        vector<double> readTargets(const vector<string>&);
        
    public:
        // Constructors
        TabularData(const string&);
        TabularData();

        // Methods
        void readTrain(string, size_t, bool header = false) override;
        void readTest(string, size_t, bool header = false) override;
        void readTrain(string, const string&) override;
        void readTest(string, const string&) override;
        // void readAllData(string, int, float);
        const Tensor& getTrainFeatures() const override;
        const Tensor& getTestFeatures() const override;
        const vector<double>& getTrainTargets() const override;
        const vector<double>& getTestTargets() const override;
        size_t getNumTrainSamples() const override;
        void headTrain(size_t numRows = 6) const override;
        void headTest(size_t numRows = 6) const override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
};