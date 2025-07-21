#pragma once

#include <cstdint>
#include <vector>
#include "core/gpu/MetalBuffer.h"

class Matrix;

using namespace std;

// Structs
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
        #ifdef __APPLE__
            MetalBuffer dataGpu;
        #endif

        // Methods
        void ensureGpu();

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

        // Methods
        Tensor& operator =(const Tensor&);

        const vector<size_t>& getShape() const;
        const vector<float>& getFlat() const;
        vector<float>& getFlat();
        size_t getSize() const;
        size_t getRank() const;

        void reduceSumBias(Tensor&) const;
        Matrix M() const;

        WindowDims computeInputWindow(size_t, size_t, Tensor::Paddings, size_t) const;
        WindowDims computeGradWindow(size_t, size_t, size_t, size_t, size_t, const WindowDims&) const;

        const Tensor& padIfNeeded(Tensor&, const WindowDims&, Tensor::Paddings, float padVal = 0.0f) const;
        void conv2dForward(const Tensor&, size_t, Tensor&, const Tensor&) const;
        void conv2dWeights(const Tensor&, size_t, size_t, size_t, size_t, Tensor&) const;
        void conv2dInput(const Tensor&, Tensor&) const;
        void padWindowInput(Tensor&, const WindowDims&, float padVal = 0.0f) const;
        void padAndUpsampleGrad(Tensor&, const WindowDims&, size_t) const;
        void maxPool2d(const WindowDims&, vector<size_t>&, size_t, size_t, size_t, Tensor::Paddings, Tensor&) const;
        void maxPool2dGrad(const Tensor&, const vector<size_t>&, Tensor&) const;

        void hadamard(const Tensor&);
        void applyGrad(const Tensor&, float);

        void reShapeInPlace(const vector<size_t>&);

        void print(const string&) const;

        // Static methods
        static Paddings decodePadding(const string&);

        // Gpu Interface
        #ifdef __APPLE__
            void initGpuTensor();
            void uploadToGpu();
            void downloadFromGpu();
        #endif

        #ifdef __OBJC__
            id<MTLBuffer> getGpuData();
            const id<MTLBuffer> getGpuData() const;

            void conv2dForwardGpu(const Tensor&, size_t, Tensor&, const Tensor&, id<MTLCommandBuffer>) const;
            void padWindowInput(Tensor&, const WindowDims&, float padVal = 0.0f, id<MTLCommandBuffer>) const;
            void hadamardGpu(const Tensor&, id<MTLCommandBuffer>);
            void applyGradGpu(const Tensor&, float, id<MTLCommandBuffer>);
        #endif
};