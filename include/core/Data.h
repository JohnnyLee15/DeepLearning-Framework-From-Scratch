#pragma once

#include <string>
#include <vector>
#include <random>

using namespace std;

class Data {
    private:
        // Instances Variables
        vector<vector<double> > trainFeatures;
        vector<int> trainTarget;
        vector<vector<double> > testFeatures;
        vector<int> testTarget;
        bool isDataLoaded;
        static const double MAX_GREYSCALE_VALUE;

        // Static Variables
        static random_device rd;
        static mt19937 generator;

        // Private Methods
        void readData(string, bool, int);
        void minmaxData();
        void minmaxNormalizeColumn(vector<vector<double> >&, double, double, int);
        void getMinMaxColumn(vector<vector<double> >&, double&, double&, int);
        void checkFile(string);
        void parseLine(string, vector<vector<double> >&, vector<int>&, int);
        void setData(vector<vector<double> >&, vector<int>&, bool);
        void normalizeGreyScale(vector<vector<double> >&); 
    
    public:
        Data();
        void readTrain(string, int);
        void readTest(string, int);
        // void readAllData(string, int, float);
        const vector<vector<double> >& getTrainFeatures() const;
        const vector<vector<double> >& getTestFeatures() const;
        const vector<int>& getTrainTarget() const;
        const vector<int>& getTestTarget() const;
        void minmax();
        void minmaxGreyScale();
        int getTrainFeatureSize() const;
        vector<int> generateShuffledIndices() const;
};