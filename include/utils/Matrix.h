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

        // Methods
        Matrix multMatMatT(const MatrixT&) const;

    public:
        // Constants
        static const int L2_CACHE_DOUBLES;

        double getValue(size_t, size_t) const;
        void setValue(size_t, size_t, double);
        void setValue(size_t, double);
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