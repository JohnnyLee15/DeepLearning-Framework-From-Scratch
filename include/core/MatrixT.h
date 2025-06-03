#pragma once

#include <vector>

class Matrix;

using namespace std;

class MatrixT {
    private:
        size_t numRows;
        size_t numCols;
        const Matrix &matrix;

    public:
        MatrixT(const Matrix&);
        size_t getNumRows() const;
        size_t getNumCols() const;
        const vector<double>& getFlat() const;

        Matrix operator *(const Matrix&) const;
        Matrix operator *(const MatrixT&) const;

};