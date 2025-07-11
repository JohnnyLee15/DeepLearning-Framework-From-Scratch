#pragma once
#include <cstdint>
#include <vector>

class Matrix;

using namespace std;

struct WindowDims {
    size_t outRows;
    size_t outCols;
    size_t padTop;
    size_t padLeft;
    size_t padRows;
    size_t padCols;
};

class Tensor {
    private:
        // Instance Variables
        vector<size_t> shape;
        vector<double> data;

    public:
        // Enums
        enum Paddings : uint32_t {
            NONE,
            SAME
        };

        // Constants
        static const string PADDING_NONE;
        static const string PADDING_SAME;

        // Constructors
        Tensor(const vector<size_t>&);
        Tensor(const vector<vector<double> >&);
        Tensor(const vector<double>&, const vector<size_t>&);
        Tensor();

        // Methods
        const vector<size_t>& getShape() const;
        const vector<double>& getFlat() const;

        vector<double>& getFlat();
        vector<double> reduceSumBias() const;

        size_t getSize() const;
        size_t getRank() const;
        Matrix M() const;

        WindowDims computeInputWindow(size_t, size_t, Tensor::Paddings, size_t) const;
        WindowDims computeGradWindow(size_t, size_t, size_t, size_t, size_t, const WindowDims&) const;

        Tensor padIfNeeded(const WindowDims&, Tensor::Paddings) const;
        Tensor conv2dForward(const Tensor&, const WindowDims&, size_t, const vector<double>& biases = {}) const;
        Tensor conv2dWeights(const Tensor&, size_t, size_t, size_t, size_t) const;
        Tensor conv2dInput(const Tensor&) const;
        Tensor reShape(const vector<size_t>&) const;
        Tensor padWindowInput(const WindowDims&) const;
        Tensor gradUpsample(size_t) const;
        Tensor maxPool2d(const WindowDims&, vector<size_t>&, size_t, size_t, size_t, Tensor::Paddings) const;
        Tensor maxPool2dGrad(const Tensor&, const vector<size_t>&) const;

        Tensor& operator *= (const Tensor&);
        Tensor& operator *= (double);
        Tensor& operator += (const Tensor&);
        // Static methods
        static Paddings decodePadding(const string&);
};