#pragma once
#include <fstream>
#include <cstdint>

class Tensor;

using namespace std;

class Layer {
    public:
        // Methods
        virtual void forward(const Tensor&) = 0;
        virtual const Tensor& getOutput() const = 0;
        virtual Tensor getOutputGradient() const = 0;
        virtual void backprop(const Tensor&, double, const Tensor&, bool) = 0;
        virtual ~Layer() = default;
        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&) = 0;
        virtual uint32_t getEncoding() const = 0;
        virtual void build(const vector<size_t>&);
        virtual vector<size_t> getBuildOutShape(const vector<size_t>&) const = 0;

        // Enums
        enum Encodings : uint32_t {
            DenseLayer,
            Conv2D,
            MaxPooling2D,
            Flatten,
            None
        };
};