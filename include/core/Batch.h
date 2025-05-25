#include <vector>

class CrossEntropy;

using namespace std;

class Batch {
    private:
        vector<vector<double> >  outputGradients;
        vector<vector<vector<double> > > layerActivations;
        vector<vector<vector<double> > > layerPreActivations;
        vector<int> indices;
        vector<vector<double> > data;
        vector<int> labels;
        int writeActivationIdx;
        int writePreActivationIdx;

    public:
        Batch(int, int);
        void setBatch(const vector<vector<double> >&, const vector<int> &);
        void setBatchIndices(int, int, const vector<int>&);
        const vector<vector<double> >& getData() const;
        void addLayerActivations(const vector<vector<double> >&);
        void addLayerPreActivations(const vector<vector<double> >&);
        const vector<vector<double> >& getLayerActivation(int) const;
        const vector<vector<double> >& getLayerPreActivation(int) const;
        const vector<vector<double> >& getOutputGradients() const;
        void updateOutputGradients(const vector<vector<double> >&);
        void calculateOutputGradients(const vector<vector<double> >&, CrossEntropy*);
        double calculateBatchLoss(const vector<vector<double> >&, CrossEntropy*);
        void writeBatchPredictions(vector<int>&, const vector<vector<double> >&) const;
        int getCorrectPredictions(const vector<int>&) const; 
};