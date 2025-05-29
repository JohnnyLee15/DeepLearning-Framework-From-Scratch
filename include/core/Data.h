#pragma once

#include <string>
#include <vector>
#include <random>
#include "utils/Matrix.h"

using namespace std;

class Data {
    private:
        // Instances Variables
        Matrix trainFeatures;
        vector<int> trainTarget;
        Matrix testFeatures;
        vector<int> testTarget;
        bool isDataLoaded;
        static const double MAX_GREYSCALE_VALUE;

        // Static Variables
        static random_device rd;
        static mt19937 generator;

        // Private Methods
        void readData(string, bool, int);
        void minmaxData();
        void minmaxNormalizeColumn(Matrix&, double, double, int);
        void getMinMaxColumn(const Matrix&, double&, double&, int);
        void checkFile(const string&);
        void parseLine(const string&, vector<double>&, int&, int);
        void setData(const Matrix&, vector<int>&, bool);
        void normalizeGreyScale(Matrix&); 
        void collectLines(vector<string>&, string);
    
    public:
        Data();
        void readTrain(string, int);
        void readTest(string, int);
        // void readAllData(string, int, float);
        const Matrix& getTrainFeatures() const;
        const Matrix& getTestFeatures() const;
        const vector<int>& getTrainTarget() const;
        const vector<int>& getTestTarget() const;
        void minmax();
        void minmaxGreyScale();
        size_t getTrainFeatureSize() const;
        vector<int> generateShuffledIndices() const;
};