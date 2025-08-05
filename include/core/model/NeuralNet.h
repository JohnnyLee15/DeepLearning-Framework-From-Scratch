#pragma once

#include <vector>
#include "core/layers/Layer.h"
#include <fstream>
#include <random>
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuTypes.h"

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
        void build(size_t, const Tensor&, bool isInference = false);

        float runEpoch(const Tensor&, const vector<float>&, float, size_t, ProgressMetric&);
        void forwardPass(Batch&);
        void forwardPassInference(const Tensor&);
        void backprop(Batch&, float);
        
        void fitBatch(Batch&, float);
        Batch makeBatch(size_t, size_t, const Tensor&, const vector<float>&, const vector<size_t>&) const;

        void loadLoss(ifstream&);
        void loadLayer(ifstream&);

        vector<size_t> generateShuffledIndices(const Tensor&) const;

        void reShapeDL(size_t);

        // GPU Interface
        #ifdef __APPLE__
            void fitBatchGpu(Batch&, float);
            void forwardPassGpu(Batch&, GpuCommandBuffer);
            void backpropGpu(Batch&, float, GpuCommandBuffer);
            void forwardPassInferenceGpu(const Tensor&);
        #endif

    public:
        // Constructors
        NeuralNet(vector<Layer*>, Loss*);
        NeuralNet();

        // Destructor
         ~NeuralNet();

        //Methods
        void fit(const Tensor&, const vector<float>&, float, float, size_t, size_t, ProgressMetric&);
        Tensor predict(const Tensor&);

        void writeBin(ofstream&) const;
        void loadFromBin(ifstream&);
};