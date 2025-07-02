#pragma once
#include <vector>
#include "core/Layer.h"
#include "utils/EpochStats.h"

class Loss;
class Activation;
class Batch;
class Tensor;
class Data;

using namespace std;

class NeuralNet {
    private:
        // Instance Variables
        vector<Layer*> layers;
        vector<double> avgLosses;
        Loss *loss;

        // Methods
        void backprop(Batch&, double);
        void forwardPass(Batch&);
        void forwardPassInference(const Tensor&);
        void updateEpochStats(EpochStats&, const Data&, const Batch&, const vector<double>&, size_t) const;
        double runEpoch(const Data&, double, vector<double>&, size_t);
        double processBatch(const Data&, Batch&, vector<double>&);
        Batch makeBatch(size_t, size_t, const Data&, const vector<size_t>&) const;
        EpochStats initEpochStats(const Data&) const;

    public:
        // Constructor
        NeuralNet(vector<Layer*>, Loss*);

        //Methods
        void train(const Data&, double, double, size_t, size_t);
        const vector<Layer*>& getLayers() const;
        const Loss* getLoss() const;
        Tensor predict(const Data&);
        void saveToBin(const string&, const Data&) const;
        ~NeuralNet();

        // Static Methods
        static NeuralNet loadFromBin(const string&, Data&);
};