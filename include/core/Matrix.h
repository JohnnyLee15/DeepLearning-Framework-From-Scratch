#pragma once

#include <vector>
#include <string>

class MatrixT;

using namespace std;

class Matrix {
    private:
    
        // Instance Variables
        vector<double> matrix;
        size_t numRows;
        size_t numCols;
        
    public:
        // Constructors        
        Matrix(size_t, size_t);
        Matrix();
        Matrix(const vector<vector<double> >&);

        // Methods
        size_t getNumCols() const;
        size_t getNumRows() const;
        Matrix operator *(const Matrix&) const;
        Matrix operator *(const MatrixT&) const;
        Matrix& operator *=(double);
        Matrix& operator *=(const Matrix&); 
        Matrix& operator += (const Matrix&);
        MatrixT T() const;
        vector<double>& getFlat();
        vector<double> operator *(const vector<double>&) const;
        vector<double> colSums() const;
        const vector<double>& getFlat() const;
        void addToRows(const vector<double>&);

        // Static Methods
        static void checkSizeMatch(size_t, size_t);
        static void checkSameShape(size_t, size_t, size_t, size_t, const string&);
        
};