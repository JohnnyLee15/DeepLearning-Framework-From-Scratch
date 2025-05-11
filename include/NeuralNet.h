#pragma once
#include <vector>
#include "Layer.h"

using namespace std;

class NeuralNet {
    private:
        // Instance Variables
        vector<Layer> layers;
        vector<double> outputActivations;
        vector<double> avgLosses;

        // Constants
        static const double LOSS_EPSILON;
        static const double GRADIENT_THRESHOLD;
        static const int PROGRESS_BAR_LENGTH;
        
        // Methods
        void applySoftmax();
        void backprop(int, double, const vector<double>&);
        void updateLayerParameters(Layer&, const vector<double>&, double, const vector<double>&);
        void forwardPass(const vector<double>&);
        void reportEpochProgress(int, int, double, double) const;
        void printProgressBar(int, int) const;

        double updateOutputDerivative(const Layer&, const vector<double>&, int);
        int getPrediction();
        double getAccurary(const vector<double>&, const vector<int>&) const;
        double clipDerivative(double);
        double runEpoch(const vector<vector<double> >&, const vector<double>&, double, vector<int>&, const vector<int>&);
        double calculateLoss(int);

        vector<double> calculateOutputGradient(int);
        vector<double> updateOutputGradient(const Layer&, const  vector<double>&, const vector<double>&);
        vector<double> getPrevActivations(int, const vector<double> &) const;
        void shuffleTrainIndices(vector<int>&) const;
        vector<int> generateIndices(const vector<vector<double> > &) const;

    public:
        NeuralNet(vector<int>);
        void train(const vector<vector<double> >&, const vector<double>&, double, double, int);
        double test(const vector<vector<double> >&, const vector<double>&);
};