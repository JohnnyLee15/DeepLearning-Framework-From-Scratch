#pragma once
#include <fstream>
#include <cstdint>

class Tensor;

using namespace std;

class Layer {
    private:
        // Instance Variables
        size_t maxBatchSize;

    public:
        // Constructor
        Layer();

        // Methods
        virtual void forward(const Tensor&) = 0;
        virtual Tensor& getOutput() = 0;
        virtual Tensor& getOutputGradient() = 0;
        virtual void backprop(const Tensor&, float, Tensor&, bool) = 0;
        virtual ~Layer() = default;
        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&) = 0;
        virtual uint32_t getEncoding() const = 0;
        virtual void build(const vector<size_t>&);
        virtual vector<size_t> getBuildOutShape(const vector<size_t>&) const = 0;
        virtual void downloadOutputFromGpu();

        size_t getMaxBatchSize() const;

        // Enums
        enum Encodings : uint32_t {
            DenseLayer,
            Conv2D,
            MaxPooling2D,
            Flatten,
            None
        };

        #ifdef __OBJC__
            virtual void forwardGpu(const Tensor&) = 0;
        #endif
};