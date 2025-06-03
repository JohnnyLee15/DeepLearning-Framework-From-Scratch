#pragma once

#include <vector>

class Matrix;

using namespace std;

class TransformUtils {
    private:
        static const double MAX_GREYSCALE_VALUE;

    public:
        static void minmaxData(Matrix&, Matrix&);
        static void minmaxNormalizeColumn(Matrix&, double, double, int);
        static void getMinMaxColumn(const Matrix&, double&, double&, int);
        static void normalizeGreyScale(Matrix&); 
};