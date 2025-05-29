#pragma once

#include <vector>
#include "utils/Matrix.h"

class MatrixT {
    private:
        size_t numRows;
        size_t numCols;
        const Matrix &matrix;
        Matrix multMatTMat(const Matrix&) const;
        Matrix multMatTMatT(const MatrixT&) const;

    public:
        MatrixT(const Matrix&);
        size_t getNumRows() const;
        size_t getNumCols() const;
        const vector<double>& getFlat() const;
        double getValue(size_t, size_t) const;

        Matrix operator *(const Matrix&) const;
        Matrix operator *(const MatrixT&) const;
        Matrix transpose() const;

};