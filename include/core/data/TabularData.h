#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "core/tensor/Tensor.h"
#include "core/data/Data.h"

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
        vector<float> trainTargets;
        vector<float> testTargets;

        vector<bool> isCategorical;
        vector<unordered_map<string, float> > featureEncodings;

        unordered_map<string, int> labelMap;

        bool isTrainLoaded;
        bool isTestLoaded;
        bool isLoadedFromModel;

        string task;

        // Methods
        void readCsv(const string&, bool, size_t, const string&, bool);
        void setData(const Tensor&, vector<float>&, bool);
        void head(size_t, const Tensor&) const;
        void checkTrainLoaded() const;
        void checkTestLoaded() const;
        void parseRawData(vector<vector<string> >&, vector<string>&, const vector<string>&, size_t);
        size_t getColIdx(const string&) const;
        vector<string> validateAndLoadCsv(const string&, bool);
        Tensor readFeatures(const vector<vector<string> >&);
        vector<float> readTargets(const vector<string>&);
        
    public:
        // Constructors
        TabularData(const string&);
        TabularData();

        // Methods
        void readTrain(const string&, size_t, bool header = false);
        void readTest(const string&, size_t, bool header = false);
        void readTrain(const string&, const string&);
        void readTest(const string&, const string&);
        // void readAllData(string, int, float);
        const Tensor& getTrainFeatures() const override;
        const Tensor& getTestFeatures() const override;
        const vector<float>& getTrainTargets() const override;
        const vector<float>& getTestTargets() const override;
        size_t getNumTrainSamples() const override;
        void headTrain(size_t numRows = 6) const;
        void headTest(size_t numRows = 6) const;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
};