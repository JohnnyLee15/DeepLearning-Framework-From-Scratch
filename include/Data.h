#include <string>
#include <vector>

using namespace std;

class Data {
    private:
        // Instances Variables
        vector<vector<double> > trainFeatures;
        vector<double> trainTarget;
        vector<vector<double> > testFeatures;
        vector<double> testTarget;
        bool isDataLoaded;

        // Private Methods
        void readData(string, bool, int);
        void getMinAndMax(double&, double&) const;
        void minmaxNormalize(double, double);
        void checkFile(string);
        void parseLine(string, vector<vector<double> >&, vector<double>&, int);
        void setData(vector<vector<double> >&, vector<double>&, bool);
    
    public:
        Data();
        void readTrain(string, int);
        void readTest(string, int);
        // void readAllData(string, int, float);
        vector<vector<double> > getTrainFeatures() const;
        vector<vector<double> > getTestFeatures() const;
        vector<double> getTrainTarget() const;
        vector<double> getTestTarget() const;
        void minmax();
};