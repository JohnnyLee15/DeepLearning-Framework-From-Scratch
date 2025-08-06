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
        // Constants
        static const size_t INFERENCE_BATCH_SIZE;

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
        void forwardPass(const Tensor&);
        void backprop(const Batch&, float);
        
        void fitBatch(const Batch&, float);
        Batch makeBatch(size_t, size_t, const Tensor&, const vector<float>&, const vector<size_t>&) const;

        void loadLoss(ifstream&);
        void loadLayer(ifstream&);

        vector<size_t> generateShuffledIndices(const Tensor&) const;

        void reShapeDL(size_t);

        Tensor makeInferenceBatch(size_t, size_t, size_t, const Tensor&) const;
        void forwardPassInference(const Tensor&);
        void cpyBatchToOutput(size_t, size_t, size_t, size_t, const Tensor&, Tensor&) const;
        // GPU Interface
        #ifdef __APPLE__
            void fitBatchGpu(const Batch&, float);
            void forwardPassGpu(const Tensor&, GpuCommandBuffer);
            void backpropGpu(const Batch&, float, GpuCommandBuffer);
            void forwardPassGpuSync(const Tensor&);
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