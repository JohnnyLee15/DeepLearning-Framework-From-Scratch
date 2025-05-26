#pragma once
#include <vector>
#include "Layer.h"

class CrossEntropy;
class Activation;
class Batch;

using namespace std;

class Data;

class NeuralNet {
    private:
        // Instance Variables
        vector<Layer> layers;
        vector<double> avgLosses;
        CrossEntropy *loss;

        // Methods
        void backprop(Batch&, double);
        void updateOutputGradients(Batch&, int);
        void forwardPass(Batch&);
        void forwardPassInference(const vector<vector<double> >&);
        double runEpoch(Data&, double, vector<int>&, int);
        double processBatch(Batch&, int, vector<int>&);
        Batch makeBatch(int, int, const vector<int>&, const vector<int>&, const vector<vector<double> >&) const;

    public:
        NeuralNet(const vector<int>&, const vector<Activation*>&, CrossEntropy*);
        void train(Data, double, double, int, int);
        double test(const vector<vector<double> >&, const vector<int>&);
        ~NeuralNet();
};