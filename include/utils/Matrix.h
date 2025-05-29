#pragma once

#include <vector>

class MatrixT;

using namespace std;

class Matrix {
    private:
    
        // Instance Variables
        vector<double> matrix;
        size_t numRows;
        size_t numCols;

    public:
        size_t getNumCols() const;
        size_t getNumRows() const;

        Matrix(size_t, size_t);
        Matrix();
        Matrix(const vector<vector<double> >&);

        Matrix operator *(const Matrix&) const;
        Matrix operator *(const MatrixT&) const;
        vector<double> operator *(const vector<double>&) const;
        Matrix& operator *=(double);
        Matrix& operator *=(const Matrix&); 
        Matrix& operator += (const Matrix&);

        const vector<double>& getFlat() const;
        vector<double>& getFlat();

        MatrixT T() const;
        vector<double> colSums() const;
        void addToRows(const vector<double>&);
        
};