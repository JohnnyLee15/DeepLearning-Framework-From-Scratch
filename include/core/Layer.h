#pragma once

#include <vector>
#include "Neuron.h"

class Activation; 

using namespace std;

class Layer {
    private:
        vector<Neuron> neurons;
        vector<double> activations;
        vector<double> preActivations;
        Activation *activation;

    public:
        Layer(int, int, Activation*);
        void calActivations(const vector<double>&);
        const vector<double>& getActivations() const;
        const vector<double>& getPreActivations() const;
        Activation* getActivation() const;
        int getNumNeurons() const;
        void updateLayerParameters(const vector<double>&, double, const vector<double>&);
        vector<double> updateOutputGradient(const  vector<double>&, const vector<double>&, Activation*);
        double updateOutputDerivative(const vector<double>&, int);
        ~Layer();
};