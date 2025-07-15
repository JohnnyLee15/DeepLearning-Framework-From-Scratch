#pragma once
#include <vector>
#include "core/Layer.h"
#include <fstream>
#include <random>
#include "core/Tensor.h"

class Loss;
class Activation;
class Batch;
class ProgressMetric;

using namespace std;

class NeuralNet {
    private:
        // Instance Variables
        vector<Layer*> layers;
        vector<float> avgLosses;
        Loss *loss;
        size_t maxBatchSize;
        Tensor dL;

        // Static variables;
        static random_device rd;
        static mt19937 generator;

        // Methods
        void backprop(Batch&, float);
        void forwardPass(Batch&);
        void forwardPassInference(const Tensor&);
        void build(size_t, const Tensor&);
        float runEpoch(const Tensor&, const vector<float>&, float, size_t, ProgressMetric&);
        Batch makeBatch(size_t, size_t, const Tensor&, const vector<float>&, const vector<size_t>&) const;
        void loadLoss(ifstream&);
        void loadLayer(ifstream&);
        vector<size_t> generateShuffledIndices(const Tensor&) const;
        void reShapeDL(size_t);

    public:
        // Constructors
        NeuralNet(vector<Layer*>, Loss*);
        NeuralNet();

        //Methods
        void fit(const Tensor&, const vector<float>&, float, float, size_t, size_t, ProgressMetric&);
        const vector<Layer*>& getLayers() const;
        const Loss* getLoss() const;
        Tensor predict(const Tensor&);
        void writeBin(ofstream&) const;
        void loadFromBin(ifstream&);
        ~NeuralNet();

        
};