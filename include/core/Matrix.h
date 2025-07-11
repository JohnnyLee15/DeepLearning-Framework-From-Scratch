#pragma once

#include <vector>
#include <string>
#include <core/Tensor.h>

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
        Tensor operator *(const Matrix&) const;
        Tensor operator *(const MatrixT&) const;
        MatrixT T() const;
        vector<double> operator *(const vector<double>&) const;
        vector<double> colSums() const;
        void addToRows(const vector<double>&);
        const vector<double>& getFlat() const;

        // Static Methods
        static void checkSizeMatch(size_t, size_t);
        static void checkSameShape(size_t, size_t, size_t, size_t, const string&);
        
};