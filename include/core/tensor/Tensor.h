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
        void padWindowInput(Tensor&, const WindowDims&, float) const;
        void padAndUpsampleGrad(Tensor&, const WindowDims&, size_t) const;
        void maxPool2d(vector<size_t>&, size_t, size_t, size_t, Tensor&) const;
        void maxPool2dGrad(const vector<size_t>&, Tensor&) const;

        void hadamard(const Tensor&);
        void applyGrad(const Tensor&, float);

        void reShapeInPlace(const vector<size_t>&);

        void zero();

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
            
            void reduceSumBiasGpu(Tensor&, id<MTLCommandBuffer>) const;
            void applyBiasGrad(Tensor&, float, id<MTLCommandBuffer>) const;

            const Tensor& padIfNeededGpu(Tensor&, const WindowDims&, Tensor::Paddings, id<MTLCommandBuffer>, float padVal = 0.0f) const;
            void padWindowInputGpu(Tensor&, const WindowDims&, id<MTLCommandBuffer>) const;
            void maxPool2dGpu(MetalBuffer&, size_t, size_t, size_t, Tensor&, id<MTLCommandBuffer>) const;

            void hadamardGpu(const Tensor&, id<MTLCommandBuffer>);
            void applyGradGpu(const Tensor&, float, id<MTLCommandBuffer>);

            void copyGpu(Tensor&, id<MTLCommandBuffer>) const;

            void conv2dForwardGpu(const Tensor&, size_t, Tensor&, const Tensor&, id<MTLCommandBuffer>) const;
            bool setConv2dForwardPipe(id<MTLComputeCommandEncoder>, uint32_t, uint32_t, uint32_t, uint32_t) const;
            void setConv2dForwardThreads(id<MTLComputeCommandEncoder>, bool, uint32_t, uint32_t, uint32_t) const;

            void conv2dWeightsGpu(const Tensor&, size_t, size_t, size_t, size_t, Tensor&, id<MTLCommandBuffer>) const;
    
            void conv2dInputGpu(const Tensor&, Tensor&, id<MTLCommandBuffer>) const;

            void maxPool2dGradGpu(MetalBuffer&, Tensor&, id<MTLCommandBuffer>) const;

            void padAndUpsampleGradGpu(Tensor&, const WindowDims&, size_t, id<MTLCommandBuffer>) const;
        #endif
};