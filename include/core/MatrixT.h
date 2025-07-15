#pragma once

#include <vector>
#include "core/GpuEngine.h"

class Matrix;
class Tensor;

using namespace std;

class MatrixT {
    private:
        // Instance Variables
        size_t numRows;
        size_t numCols;
        const Matrix &matrix;

    public:
        // Constructor
        MatrixT(const Matrix&);

        // Methods
        size_t getNumRows() const;
        size_t getNumCols() const;
        const vector<float>& getFlat() const;
        void mTm(const Matrix&, Tensor&) const;
        void mTmT(const MatrixT&, Tensor&) const;

        // GPU methods
        #ifdef __OBJC__
            void mTmGpu(const Matrix&, Tensor&, id<MTLCommandBuffer>) const;
            void mTmTGpu(const MatrixT&, Tensor&, id<MTLCommandBuffer>) const;
            id<MTLBuffer> getGpuData() const;
        #endif

};