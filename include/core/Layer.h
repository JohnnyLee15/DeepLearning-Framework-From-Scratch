#pragma once
#include <fstream>

class Matrix;

using namespace std;

class Layer {
    public:
        // Methods
        virtual void calActivations(const Matrix&) = 0;
        virtual const Matrix getActivations() const = 0;
        virtual Matrix getOutputGradient() const = 0;
        virtual void backprop(const Matrix&, double, const Matrix&, bool) = 0;
        virtual ~Layer() = default;
        virtual void writeBin(ofstream&) const = 0;
        virtual void loadWeightsAndBiases(ifstream&) = 0;

        // Enums
        enum Encodings : uint32_t {
            DenseLayer
        };
};