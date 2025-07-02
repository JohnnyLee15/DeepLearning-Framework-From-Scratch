#pragma once

#include <vector>

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
        const vector<double>& getFlat() const;
        Tensor operator *(const Matrix&) const;
        Tensor operator *(const MatrixT&) const;

};