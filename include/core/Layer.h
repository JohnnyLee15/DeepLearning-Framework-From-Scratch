#pragma once

#include <vector>

class Activation; 

using namespace std;

class Layer {
    private:

        // Constants
        static const double HE_INT_GAIN;

        // Instance Variables
        vector<vector<double> > activations;
        vector<vector<double> > preActivations;
        Activation *activation;
        vector<vector<double> > weights;
        vector<double> biases;

        // Methods
        vector<vector<double> > getActivationGradientMat(const vector<vector<double> >&, Activation*) const;
        void initWeights();
        void initBiases();

    public:
        Layer(int, int, Activation*);
        void calActivations(const vector<double>&);
        void calActivations(const vector<vector<double> >&);
        const vector<vector<double> > getActivations() const;
        const vector<vector<double> > getPreActivations() const;
        Activation* getActivation() const;
        void updateLayerParameters(const vector<vector<double> >&, double, const vector<vector<double> >&);
        vector<vector<double> > updateOutputGradient(const vector<vector<double> >&, const vector<vector<double> >&, Activation*);
        ~Layer();
};