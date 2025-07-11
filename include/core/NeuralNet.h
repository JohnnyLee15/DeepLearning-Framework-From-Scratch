#pragma once
#include <vector>
#include "core/Layer.h"
#include <fstream>
#include <random>

class Loss;
class Activation;
class Batch;
class Tensor;
class ProgressMetric;

using namespace std;

class NeuralNet {
    private:
        // Instance Variables
        vector<Layer*> layers;
        vector<double> avgLosses;
        Loss *loss;

        // Static variables;
        static random_device rd;
        static mt19937 generator;

        // Methods
        void backprop(Batch&, double);
        void forwardPass(Batch&);
        void forwardPassInference(const Tensor&);
        void build(const Tensor&);
        double runEpoch(const Tensor&, const vector<double>&, double, size_t, ProgressMetric&);
        Batch makeBatch(size_t, size_t, const Tensor&, const vector<double>&, const vector<size_t>&) const;
        void loadLoss(ifstream&);
        void loadLayer(ifstream&);
        vector<size_t> generateShuffledIndices(const Tensor&) const;

    public:
        // Constructors
        NeuralNet(vector<Layer*>, Loss*);
        NeuralNet();

        //Methods
        void fit(const Tensor&, const vector<double>&, double, double, size_t, size_t, ProgressMetric&);
        const vector<Layer*>& getLayers() const;
        const Loss* getLoss() const;
        Tensor predict(const Tensor&);
        void writeBin(ofstream&) const;
        void loadFromBin(ifstream&);
        ~NeuralNet();

        
};