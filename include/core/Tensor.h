#pragma once

#include <vector>

class Matrix;

using namespace std;

class Tensor {
    private:
        vector<size_t> shape;
        vector<double> data;

    public:
        Tensor(const vector<size_t>&);
        Tensor(const vector<vector<double> >&);
        Tensor();
        const vector<size_t>& getShape() const;
        const vector<double>& getFlat() const;
        vector<double>& getFlat();
        size_t getSize() const;
        size_t getRank() const;
        Matrix M() const;

};