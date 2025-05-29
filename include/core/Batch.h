#include <vector>
#include "utils/Matrix.h"

class CrossEntropy;

using namespace std;

class Batch {
    private:
        size_t batchSize;
        Matrix outputGradients;
        vector<Matrix> layerActivations;
        vector<Matrix> layerPreActivations;
        vector<int> indices;
        Matrix data;
        vector<int> labels;
        int writeActivationIdx;
        int writePreActivationIdx;

    public:
        Batch(int, int);
        void setBatch(const Matrix&, const vector<int> &);
        void setBatchIndices(int, int, const vector<int>&);
        const Matrix& getData() const;
        void addLayerActivations(const Matrix&);
        void addLayerPreActivations(const Matrix&);
        const Matrix& getLayerActivation(int) const;
        const Matrix& getLayerPreActivation(int) const;
        const Matrix& getOutputGradients() const;
        void updateOutputGradients(const Matrix&);
        void calculateOutputGradients(const Matrix&, CrossEntropy*);
        double calculateBatchLoss(const Matrix&, CrossEntropy*);
        void writeBatchPredictions(vector<int>&, const Matrix&) const;
        int getCorrectPredictions(const vector<int>&) const; 
};