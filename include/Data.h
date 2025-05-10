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
        static const double MAX_GREYSCALE_VALUE;

        // Private Methods
        void readData(string, bool, int);
        void minmaxData();
        void minmaxNormalizeColumn(vector<vector<double> >&, double, double, int);
        void getMinMaxColumn(vector<vector<double> >&, double&, double&, int);
        void checkFile(string);
        void parseLine(string, vector<vector<double> >&, vector<double>&, int);
        void setData(vector<vector<double> >&, vector<double>&, bool);
        void normalizeGreyScale(vector<vector<double> >&); 
    
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
        void minmaxGreyScale();
};