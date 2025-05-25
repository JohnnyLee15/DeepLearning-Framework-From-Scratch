#include "utils/MatrixUtils.h"
#include <cassert>
#include <omp.h>

vector<double> MatrixUtils::multMatVec(
    const vector<vector<double> > &mat,
    const vector<double> &vec
) {
    assert(mat[0].size() == vec.size());

    int matRows = mat.size();
    vector<double> product(matRows, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < matRows; i++) {
        product[i] = dot(mat[i], vec);
    }

    return product;
}

double MatrixUtils::dot(
    const vector<double> &vec1,
    const vector<double> &vec2
) {
    assert(vec1.size() == vec2.size());

    int size = vec1.size();
    double product = 0.0;

    #pragma omp parallel for reduction(+:product)
    for (int i = 0; i < size; i++) {
        product += vec1[i] * vec2[i];
    }

    return product;
}

void MatrixUtils::addVecInplace(
    vector<double> &vec1,
    const vector<double> &vec2
) {
    assert(vec1.size() == vec2.size());

    int size = vec1.size();

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        vec1[i] += vec2[i];
    }
}

vector<vector<double> > MatrixUtils::multMatTMat(
    const vector<vector<double> > &mat1,
    const vector<vector<double> > &mat2
) {
    int mat1TRows = mat1[0].size();
    int mat1TCols = mat1.size();
    int mat2Rows = mat2.size();
    int mat2Cols = mat2[0].size();

    assert(mat1TCols == mat2Rows);

    vector<vector<double> > product(mat1TRows, vector<double>(mat2Cols, 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat1TRows; i++) {
        for (int j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (int k = 0; k < mat2Rows; k++) {
                value += mat1[k][i] * mat2[k][j];
            }
            product[i][j] = value;
        }
    }

    return product;
}

vector<vector<double> > MatrixUtils::multMatMatT(
    const vector<vector<double> > &mat1,
    const vector<vector<double> > &mat2
) {
    int mat1Rows = mat1.size();
    int mat1Cols = mat1[0].size();
    int mat2TRows = mat2[0].size();
    int mat2TCols = mat2.size();

    assert(mat1Cols == mat2TRows);

    vector<vector<double> > product(mat1Rows, vector<double>(mat2TCols, 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat1Rows; i++) {
        for (int j = 0; j < mat2TCols; j++) {
            double value = 0.0;
            for (int k = 0; k < mat2TRows; k++) {
                value += mat1[i][k] * mat2[j][k];
            }
            product[i][j] = value;
        }
    }

    return product;
}

void MatrixUtils::scaleVecInplace(
    vector<double> &vec,
    double scaleFactor
) {
    int size = vec.size();

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        vec[i] *= scaleFactor;
    }
}

vector<double> MatrixUtils::colSums(
    const vector<vector<double> > &mat
) {
    int numCols = mat[0].size();
    int numRows = mat.size(); 
    vector<double> colSumsVec(numCols, 0.0);

    #pragma omp parallel 
    {
        vector<double> threadColSums(numCols, 0.0);

        #pragma omp for
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                threadColSums[j] += mat[i][j];
            }
        }
        
        #pragma omp critical 
        {
            for (int j = 0; j < numCols; j++) {
                colSumsVec[j] += threadColSums[j];
            }
        }
    }

    return colSumsVec;
}

vector<vector<double> > MatrixUtils::multMatMat(
    const vector<vector<double> > &mat1,
    const vector<vector<double> > &mat2
) {
    int mat1Cols = mat1[0].size();
    int mat1Rows = mat1.size();
    int mat2Cols = mat2[0].size();
    int mat2Rows = mat2.size();
    assert(mat1Cols == mat2Rows);

    vector<vector<double> > product(mat1Rows, vector<double>(mat2Cols, 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat1Rows; i++) {
        for (int j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (int k = 0; k < mat2Rows; k++) {
                value += mat1[i][k] * mat2[k][j];
            }
            product[i][j] = value;
        }
    }

    return product;
}

void MatrixUtils::hardamardInplace(
    vector<vector<double> > &mat1, 
    const vector<vector<double> > &mat2
) {
    assert(mat1[0].size() == mat2[0].size());
    assert(mat1.size() == mat2.size());

    int numRows = mat1.size();
    int numCols = mat1[0].size();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            mat1[i][j] *= mat2[i][j];
        }
    }
}

void MatrixUtils::scaleMatInplace(
    vector<vector<double> > &mat,
    double scaleFactor
) {
    int numRows = mat.size();
    int numCols = mat[0].size();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            mat[i][j] *= scaleFactor;
        }
    }
}

void MatrixUtils::addMatInplace(
    vector<vector<double> > &mat1,
    const vector<vector<double> > &mat2
) {
    assert(mat1[0].size() == mat2[0].size());
    assert(mat1.size() == mat2.size());

    int numRows = mat1.size();
    int numCols = mat1[0].size();


    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            mat1[i][j] += mat2[i][j];
        }
    }
}