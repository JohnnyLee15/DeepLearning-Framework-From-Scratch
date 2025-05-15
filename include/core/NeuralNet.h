#pragma once
#include <vector>
#include "Layer.h"

class CrossEntropy;
class Activation;

using namespace std;

class Data;

class NeuralNet {
    private:
        // Instance Variables
        vector<Layer*> layers;
        vector<double> avgLosses;
        CrossEntropy *loss;
        
        // Methods
        void backprop(int, double, const vector<double>&);
        void forwardPass(const vector<double>&);
        double runEpoch(Data&, double, vector<int>&, const vector<int>&);
        const vector<double>& getPrevActivations(int, const vector<double> &) const;

    public:
        NeuralNet(const vector<int>&, const vector<Activation*>&, CrossEntropy*);
        void train(Data, double, double, int);
        double test(const vector<vector<double> >&, const vector<double>&);
        ~NeuralNet();
};