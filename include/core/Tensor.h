#pragma once
#include <cstdint>
#include <vector>
#include "core/GpuEngine.h"

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
        vector<float> data;
        
        // GPU Instance Variables
        #ifdef __OBJC__
            id<MTLBuffer> dataGpu;
        #endif

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
        Tensor(const vector<vector<float> >&);
        Tensor(const vector<float>&, const vector<size_t>&);
        Tensor(const Tensor&);
        Tensor();

        Tensor& operator =(const Tensor&);

        // Methods
        const vector<size_t>& getShape() const;
        const vector<float>& getFlat() const;

        void ensureGpu();

        vector<float>& getFlat();
        void reduceSumBias(Tensor&) const;

        size_t getSize() const;
        size_t getRank() const;
        Matrix M() const;

        WindowDims computeInputWindow(size_t, size_t, Tensor::Paddings, size_t) const;
        WindowDims computeGradWindow(size_t, size_t, size_t, size_t, size_t, const WindowDims&) const;

        Tensor padIfNeeded(const WindowDims&, Tensor::Paddings) const;
        Tensor conv2dForward(const Tensor&, const WindowDims&, size_t, const Tensor& biases = Tensor()) const;
        Tensor conv2dWeights(const Tensor&, size_t, size_t, size_t, size_t) const;
        Tensor conv2dInput(const Tensor&) const;
        Tensor reShape(const vector<size_t>&) const;
        Tensor padWindowInput(const WindowDims&) const;
        Tensor gradUpsample(size_t) const;
        Tensor maxPool2d(const WindowDims&, vector<size_t>&, size_t, size_t, size_t, Tensor::Paddings) const;
        Tensor maxPool2dGrad(const Tensor&, const vector<size_t>&) const;

        void hadamard(const Tensor&);
        void applyGrad(const Tensor&, float);

        void reShapeInPlace(const vector<size_t>&);

        void print(const string&) const;

        // Static methods
        static Paddings decodePadding(const string&);

        // Gpu Methods

        #ifdef __APPLE__
            void initGpuTensor();
            void uploadToGpu();
        #endif

        #ifdef __OBJC__
            id<MTLBuffer> getGpuData();
            const id<MTLBuffer> getGpuData() const;

            void downloadFromGpu();

            void hadamardGpu(const Tensor&, id<MTLCommandBuffer>);
            void applyGradGpu(const Tensor&, float, id<MTLCommandBuffer>);
        #endif
};