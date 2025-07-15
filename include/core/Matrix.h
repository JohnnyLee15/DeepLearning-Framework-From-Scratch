#pragma once

#include <vector>
#include <string>
#include <core/Tensor.h>
#include "core/GpuEngine.h"

class MatrixT;

using namespace std;

class Matrix {
    private:
        Tensor &tensor;

    public:
        // Constructors        
        Matrix(Tensor&);

        // Methods
        size_t getNumCols() const;
        size_t getNumRows() const;
        void mm(const Matrix&, Tensor&) const;
        void mmT(const MatrixT&, Tensor&) const;
        void colSums(Tensor&) const;
        void addToRows(const Tensor&);
        MatrixT T() const;

        const vector<float>& getFlat() const;

        // Static Methods
        static void checkSizeMatch(size_t, size_t);
        static void checkSameShape(size_t, size_t, size_t, size_t, const string&);


        // GPU
        #ifdef __OBJC__
            // Static Methods
            static void matMatEngine(
                id<MTLBuffer>,
                id<MTLBuffer>,
                id<MTLBuffer>,
                size_t,
                size_t,
                size_t,
                id<MTLComputePipelineState>, 
                id<MTLCommandBuffer>
            );

            // Instance Methods
            void mmGpu(const Matrix&, Tensor&, id<MTLCommandBuffer>) const;
            void mmTGpu(const MatrixT&, Tensor&, id<MTLCommandBuffer>) const;
            void colSumsGpu(Tensor&, id<MTLCommandBuffer>) const;
            void addToRowsGpu(const Tensor&, id<MTLCommandBuffer>);
            id<MTLBuffer> getGpuData() const;
        #endif
};