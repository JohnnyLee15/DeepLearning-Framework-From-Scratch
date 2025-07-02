#pragma once
#include <fstream>
#include <cstdint>

class Tensor;

using namespace std;

class Layer {
    public:
        // Methods
        virtual void calActivations(const Tensor&) = 0;
        virtual const Tensor getActivations() const = 0;
        virtual Tensor getOutputGradient() const = 0;
        virtual void backprop(const Tensor&, double, const Tensor&, bool) = 0;
        virtual ~Layer() = default;
        virtual void writeBin(ofstream&) const;
        virtual void loadWeightsAndBiases(ifstream&) = 0;
        virtual uint32_t getEncoding() const = 0;

        // Enums
        enum Encodings : uint32_t {
            DenseLayer
        };
};