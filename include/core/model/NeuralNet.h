#pragma once

#include <vector>
#include "core/layers/Layer.h"
#include <fstream>
#include <random>
#include "core/tensor/Tensor.h"
#include "core/data/BinLoader.h"
#include "core/gpu/GpuTypes.h"

class Loss;
class Activation;
class Batch;
class ProgressMetric;
class EarlyStop;
class Dataset;

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
        void build(size_t, vector<size_t>, bool isInference = false);

        void fitInternal(
            const Dataset&, const Dataset&, float, 
            float, size_t, size_t, ProgressMetric&, EarlyStop*
        );

        Tensor predictInternal(const Dataset&);

        float runEpoch(const Dataset&, float, size_t, ProgressMetric&);

        void forwardPass(const Tensor&);
        void backprop(const Batch&, float);
        
        void fitBatch(const Batch&, float);

        void loadLoss(ifstream&);
        Layer* loadLayer(ifstream&);

        vector<size_t> generateShuffledIndices(size_t) const;

        void reShapeDL(size_t);

        void forwardPassInference(const Tensor&);
        void cpyBatchToOutput(size_t, size_t, size_t, size_t, Tensor&) const;

        bool validateEpoch(const Dataset&, ProgressMetric&, EarlyStop*, size_t);

        void deleteLayers();
        void tryBestWeights(EarlyStop *stop);

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
        NeuralNet(const NeuralNet&);

        // Destructor
        ~NeuralNet();

        //Methods
        void fit(
            const Tensor&, const vector<float>&, float, 
            float, size_t, size_t, ProgressMetric&,
            const Tensor& xVal = Tensor(),
            const vector<float>& yVal = {},
            EarlyStop *stop = nullptr
        );

        void fit(
            const BinLoader&, float, 
            float, size_t, size_t, ProgressMetric&,
            const BinLoader& val = BinLoader(),
            EarlyStop *stop = nullptr
        );

        Tensor predict(const Tensor&);
        Tensor predict(const BinLoader&);

        void writeBin(ofstream&) const;
        void loadFromBin(ifstream&);
        void saveBestWeights(ofstream&) const;
        void loadBestWeights(ifstream&);

        NeuralNet* clone() const;
};